//! Statement parser.
//!
//! Handles all Python statement types including compound statements.

use crate::ast::{
    Alias, Arg, Arguments, AugOp, ExceptHandler, Expr, ExprKind, MatchCase, Pattern, PatternKind,
    Singleton, Stmt, StmtKind, TypeParam, TypeParamKind, WithItem,
};
use crate::parser::{ExprParser, Parser, Precedence};
use crate::token::{Keyword as KW, TokenKind};
use prism_core::PrismResult;

/// Statement parser.
pub struct StmtParser;

impl StmtParser {
    /// Parse a statement.
    pub fn parse(parser: &mut Parser<'_>) -> PrismResult<Stmt> {
        let start = parser.start_span();

        // Check for compound statements
        if let TokenKind::Keyword(kw) = &parser.current().kind {
            match kw {
                KW::If => return Self::parse_if(parser, start),
                KW::While => return Self::parse_while(parser, start),
                KW::For => return Self::parse_for(parser, start, false),
                KW::Try => return Self::parse_try(parser, start),
                KW::With => return Self::parse_with(parser, start, false),
                KW::Def => return Self::parse_function_def(parser, start, false),
                KW::Class => return Self::parse_class_def(parser, start),
                KW::Async => return Self::parse_async(parser, start),
                KW::Match => return Self::parse_match(parser, start),
                _ => {}
            }
        }

        // Check for decorator
        if parser.check(TokenKind::At) {
            return Self::parse_decorated(parser, start);
        }

        // Simple statement
        Self::parse_simple_statement(parser, start)
    }

    /// Parse a simple statement.
    fn parse_simple_statement(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        let stmt = match &parser.current().kind.clone() {
            TokenKind::Keyword(KW::Pass) => {
                parser.advance();
                Stmt::new(StmtKind::Pass, parser.span_from(start))
            }
            TokenKind::Keyword(KW::Break) => {
                parser.advance();
                Stmt::new(StmtKind::Break, parser.span_from(start))
            }
            TokenKind::Keyword(KW::Continue) => {
                parser.advance();
                Stmt::new(StmtKind::Continue, parser.span_from(start))
            }
            TokenKind::Keyword(KW::Return) => Self::parse_return(parser, start)?,
            TokenKind::Keyword(KW::Raise) => Self::parse_raise(parser, start)?,
            TokenKind::Keyword(KW::Assert) => Self::parse_assert(parser, start)?,
            TokenKind::Keyword(KW::Del) => Self::parse_delete(parser, start)?,
            TokenKind::Keyword(KW::Global) => Self::parse_global(parser, start)?,
            TokenKind::Keyword(KW::Nonlocal) => Self::parse_nonlocal(parser, start)?,
            TokenKind::Keyword(KW::Import) => Self::parse_import(parser, start)?,
            TokenKind::Keyword(KW::From) => Self::parse_from_import(parser, start)?,
            TokenKind::Keyword(KW::Type) => Self::parse_type_alias(parser, start)?,
            _ => Self::parse_expression_statement(parser, start)?,
        };

        // Consume trailing newline if present
        parser.match_token(TokenKind::Newline);
        Ok(stmt)
    }

    // =========================================================================
    // Simple Statements
    // =========================================================================

    /// Parse an expression statement (possibly assignment).
    fn parse_expression_statement(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        let first_start = parser.start_span();
        let first = ExprParser::parse(parser, Precedence::Lowest)?;
        let first = Self::parse_comma_tuple_expr(parser, first_start, first, true)?;

        // Check for assignment
        if parser.match_token(TokenKind::Equal) {
            let mut targets = vec![first];

            // Handle chained assignment: a = b = c = value
            loop {
                let value_start = parser.start_span();
                let value_first = ExprParser::parse(parser, Precedence::Lowest)?;
                let value = Self::parse_comma_tuple_expr(parser, value_start, value_first, true)?;
                if parser.match_token(TokenKind::Equal) {
                    targets.push(value);
                } else {
                    return Ok(Stmt::new(
                        StmtKind::Assign {
                            targets,
                            value: Box::new(value),
                        },
                        parser.span_from(start),
                    ));
                }
            }
        }

        // Check for augmented assignment
        if let Some(op) = Self::match_aug_assign(parser) {
            let value = ExprParser::parse(parser, Precedence::Lowest)?;
            return Ok(Stmt::new(
                StmtKind::AugAssign {
                    target: Box::new(first),
                    op,
                    value: Box::new(value),
                },
                parser.span_from(start),
            ));
        }

        // Check for annotated assignment
        if parser.match_token(TokenKind::Colon) {
            let annotation = ExprParser::parse(parser, Precedence::Lowest)?;
            let value = if parser.match_token(TokenKind::Equal) {
                Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
            } else {
                None
            };
            let simple = matches!(first.kind, ExprKind::Name(_));
            return Ok(Stmt::new(
                StmtKind::AnnAssign {
                    target: Box::new(first),
                    annotation: Box::new(annotation),
                    value,
                    simple,
                },
                parser.span_from(start),
            ));
        }

        // Plain expression statement
        Ok(Stmt::new(
            StmtKind::Expr(Box::new(first)),
            parser.span_from(start),
        ))
    }

    /// Parse trailing comma-separated expressions into a tuple expression.
    ///
    /// This supports Python's implicit tuple syntax in statement contexts:
    /// - Targets: `a, b = ...`
    /// - Values: `a = 1, 2`
    /// - Chained assignments: `a = b = 1, 2`
    fn parse_comma_tuple_expr(
        parser: &mut Parser<'_>,
        start: u32,
        first: Expr,
        stop_at_equal: bool,
    ) -> PrismResult<Expr> {
        if !parser.match_token(TokenKind::Comma) {
            return Ok(first);
        }

        let mut elements = vec![first];
        while !parser.check(TokenKind::Newline) && !parser.check(TokenKind::Eof) {
            if stop_at_equal && parser.check(TokenKind::Equal) {
                break;
            }

            elements.push(ExprParser::parse(parser, Precedence::Lowest)?);
            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        Ok(Expr::new(
            ExprKind::Tuple(elements),
            parser.span_from(start),
        ))
    }

    /// Match an augmented assignment operator.
    fn match_aug_assign(parser: &mut Parser<'_>) -> Option<AugOp> {
        let op = match &parser.current().kind {
            TokenKind::PlusEqual => AugOp::Add,
            TokenKind::MinusEqual => AugOp::Sub,
            TokenKind::StarEqual => AugOp::Mult,
            TokenKind::AtEqual => AugOp::MatMult,
            TokenKind::SlashEqual => AugOp::Div,
            TokenKind::DoubleSlashEqual => AugOp::FloorDiv,
            TokenKind::PercentEqual => AugOp::Mod,
            TokenKind::DoubleStarEqual => AugOp::Pow,
            TokenKind::LeftShiftEqual => AugOp::LShift,
            TokenKind::RightShiftEqual => AugOp::RShift,
            TokenKind::AmpersandEqual => AugOp::BitAnd,
            TokenKind::PipeEqual => AugOp::BitOr,
            TokenKind::CaretEqual => AugOp::BitXor,
            _ => return None,
        };
        parser.advance();
        Some(op)
    }

    /// Parse a return statement.
    fn parse_return(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'return'

        let value = if parser.check(TokenKind::Newline) || parser.check(TokenKind::Eof) {
            None
        } else {
            // Parse first expression
            let first = ExprParser::parse(parser, Precedence::Lowest)?;

            // Check for comma (tuple return)
            if parser.match_token(TokenKind::Comma) {
                let first_start = first.span.start;
                let mut elements = vec![first];

                // Parse remaining comma-separated expressions
                while !parser.check(TokenKind::Newline) && !parser.check(TokenKind::Eof) {
                    elements.push(ExprParser::parse(parser, Precedence::Lowest)?);
                    if !parser.match_token(TokenKind::Comma) {
                        break;
                    }
                }

                // Create tuple from elements
                Some(Box::new(Expr::new(
                    ExprKind::Tuple(elements),
                    parser.span_from(first_start),
                )))
            } else {
                Some(Box::new(first))
            }
        };

        Ok(Stmt::new(StmtKind::Return(value), parser.span_from(start)))
    }

    /// Parse a raise statement.
    fn parse_raise(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'raise'

        if parser.check(TokenKind::Newline) || parser.check(TokenKind::Eof) {
            return Ok(Stmt::new(
                StmtKind::Raise {
                    exc: None,
                    cause: None,
                },
                parser.span_from(start),
            ));
        }

        let exc = Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?));

        let cause = if parser.match_keyword(KW::From) {
            Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
        } else {
            None
        };

        Ok(Stmt::new(
            StmtKind::Raise { exc, cause },
            parser.span_from(start),
        ))
    }

    /// Parse an assert statement.
    fn parse_assert(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'assert'

        let test = ExprParser::parse(parser, Precedence::Lowest)?;

        let msg = if parser.match_token(TokenKind::Comma) {
            Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
        } else {
            None
        };

        Ok(Stmt::new(
            StmtKind::Assert {
                test: Box::new(test),
                msg,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a delete statement.
    fn parse_delete(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'del'

        let mut targets = Vec::new();
        targets.push(ExprParser::parse(parser, Precedence::Lowest)?);

        while parser.match_token(TokenKind::Comma) {
            targets.push(ExprParser::parse(parser, Precedence::Lowest)?);
        }

        Ok(Stmt::new(
            StmtKind::Delete(targets),
            parser.span_from(start),
        ))
    }

    /// Parse a global statement.
    fn parse_global(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'global'

        let mut names = Vec::new();
        names.push(parser.expect_identifier("expected identifier")?);

        while parser.match_token(TokenKind::Comma) {
            names.push(parser.expect_identifier("expected identifier")?);
        }

        Ok(Stmt::new(StmtKind::Global(names), parser.span_from(start)))
    }

    /// Parse a nonlocal statement.
    fn parse_nonlocal(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'nonlocal'

        let mut names = Vec::new();
        names.push(parser.expect_identifier("expected identifier")?);

        while parser.match_token(TokenKind::Comma) {
            names.push(parser.expect_identifier("expected identifier")?);
        }

        Ok(Stmt::new(
            StmtKind::Nonlocal(names),
            parser.span_from(start),
        ))
    }

    /// Parse an import statement.
    fn parse_import(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'import'

        let mut names = Vec::new();
        names.push(Self::parse_alias(parser)?);

        while parser.match_token(TokenKind::Comma) {
            names.push(Self::parse_alias(parser)?);
        }

        Ok(Stmt::new(StmtKind::Import(names), parser.span_from(start)))
    }

    /// Parse a from-import statement.
    fn parse_from_import(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'from'

        // Count leading dots for relative imports
        let mut level = 0;
        while parser.match_token(TokenKind::Dot) || parser.match_token(TokenKind::Ellipsis) {
            if parser.previous().kind == TokenKind::Ellipsis {
                level += 3;
            } else {
                level += 1;
            }
        }

        // Parse module name
        let module = if let TokenKind::Ident(_) = &parser.current().kind {
            Some(Self::parse_dotted_name(parser)?)
        } else {
            None
        };

        parser.expect_keyword(KW::Import, "expected 'import'")?;

        // Check for 'from x import *'
        if parser.match_token(TokenKind::Star) {
            return Ok(Stmt::new(
                StmtKind::ImportFrom {
                    module,
                    names: vec![Alias {
                        name: "*".to_string(),
                        asname: None,
                        span: parser.span_from(start),
                    }],
                    level,
                },
                parser.span_from(start),
            ));
        }

        // Check for 'from x import (a, b, c)'
        let has_paren = parser.match_token(TokenKind::LeftParen);

        let mut names = Vec::new();
        names.push(Self::parse_alias(parser)?);

        while parser.match_token(TokenKind::Comma) {
            if has_paren && parser.check(TokenKind::RightParen) {
                break;
            }
            names.push(Self::parse_alias(parser)?);
        }

        if has_paren {
            parser.expect(TokenKind::RightParen, "expected ')'")?;
        }

        Ok(Stmt::new(
            StmtKind::ImportFrom {
                module,
                names,
                level,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a dotted name (module path).
    fn parse_dotted_name(parser: &mut Parser<'_>) -> PrismResult<String> {
        let mut name = parser.expect_identifier("expected module name")?;
        while parser.match_token(TokenKind::Dot) {
            name.push('.');
            name.push_str(&parser.expect_identifier("expected identifier")?);
        }
        Ok(name)
    }

    /// Parse an import alias.
    fn parse_alias(parser: &mut Parser<'_>) -> PrismResult<Alias> {
        let start = parser.start_span();
        let name = Self::parse_dotted_name(parser)?;

        let asname = if parser.match_keyword(KW::As) {
            Some(parser.expect_identifier("expected identifier")?)
        } else {
            None
        };

        Ok(Alias {
            name,
            asname,
            span: parser.span_from(start),
        })
    }

    /// Parse a type alias statement (Python 3.12+).
    fn parse_type_alias(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'type'

        let name = parser.expect_identifier("expected type name")?;
        let name_span = parser.span_from(start);

        let type_params = if parser.match_token(TokenKind::LeftBracket) {
            let params = Self::parse_type_params(parser)?;
            parser.expect(TokenKind::RightBracket, "expected ']'")?;
            params
        } else {
            Vec::new()
        };

        parser.expect(TokenKind::Equal, "expected '='")?;
        let value = ExprParser::parse(parser, Precedence::Lowest)?;

        Ok(Stmt::new(
            StmtKind::TypeAlias {
                name: Box::new(Expr::new(ExprKind::Name(name), name_span)),
                type_params,
                value: Box::new(value),
            },
            parser.span_from(start),
        ))
    }

    // =========================================================================
    // Compound Statements
    // =========================================================================

    /// Parse an if statement.
    fn parse_if(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'if'

        let test = ExprParser::parse(parser, Precedence::Lowest)?;
        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        let mut orelse = Vec::new();
        while parser.match_keyword(KW::Elif) {
            let elif_start = parser.previous().span.start;
            let elif_test = ExprParser::parse(parser, Precedence::Lowest)?;
            parser.expect(TokenKind::Colon, "expected ':'")?;
            let elif_body = Self::parse_block(parser)?;
            orelse.push(Stmt::new(
                StmtKind::If {
                    test: Box::new(elif_test),
                    body: elif_body,
                    orelse: Vec::new(),
                },
                parser.span_from(elif_start),
            ));
        }

        if parser.match_keyword(KW::Else) {
            parser.expect(TokenKind::Colon, "expected ':'")?;
            let else_body = Self::parse_block(parser)?;
            // Attach else to the innermost if
            if let Some(last) = orelse.last_mut() {
                if let StmtKind::If { orelse: inner, .. } = &mut last.kind {
                    *inner = else_body;
                }
            } else {
                orelse = else_body;
            }
        }

        Ok(Stmt::new(
            StmtKind::If {
                test: Box::new(test),
                body,
                orelse,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a while statement.
    fn parse_while(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'while'

        let test = ExprParser::parse(parser, Precedence::Lowest)?;
        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        let orelse = if parser.match_keyword(KW::Else) {
            parser.expect(TokenKind::Colon, "expected ':'")?;
            Self::parse_block(parser)?
        } else {
            Vec::new()
        };

        Ok(Stmt::new(
            StmtKind::While {
                test: Box::new(test),
                body,
                orelse,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a for statement.
    fn parse_for(parser: &mut Parser<'_>, start: u32, is_async: bool) -> PrismResult<Stmt> {
        parser.advance(); // consume 'for'

        // Parse target - need to handle tuple targets like `k, v`
        // First parse at BitwiseOr to get the first element
        let target_start = parser.start_span();
        let first = ExprParser::parse(parser, Precedence::BitwiseOr)?;

        // Check for comma (tuple target like `for k, v in items`)
        let target = if parser.match_token(TokenKind::Comma) {
            let mut elements = vec![first];
            // Parse remaining comma-separated targets until 'in' keyword
            while !parser.check_keyword(KW::In) {
                elements.push(ExprParser::parse(parser, Precedence::BitwiseOr)?);
                if !parser.match_token(TokenKind::Comma) {
                    break;
                }
            }
            // Create tuple from elements
            Expr::new(ExprKind::Tuple(elements), parser.span_from(target_start))
        } else {
            first
        };

        parser.expect_keyword(KW::In, "expected 'in'")?;
        let iter = ExprParser::parse(parser, Precedence::Lowest)?;
        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        let orelse = if parser.match_keyword(KW::Else) {
            parser.expect(TokenKind::Colon, "expected ':'")?;
            Self::parse_block(parser)?
        } else {
            Vec::new()
        };

        let kind = if is_async {
            StmtKind::AsyncFor {
                target: Box::new(target),
                iter: Box::new(iter),
                body,
                orelse,
            }
        } else {
            StmtKind::For {
                target: Box::new(target),
                iter: Box::new(iter),
                body,
                orelse,
            }
        };

        Ok(Stmt::new(kind, parser.span_from(start)))
    }

    /// Parse a try statement.
    fn parse_try(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'try'
        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        let is_star = parser.check_keyword(KW::Except) && {
            // Peek to see if it's 'except*'
            false // Simplified - would need lookahead
        };

        let mut handlers = Vec::new();
        while parser.match_keyword(KW::Except) {
            let handler_start = parser.previous().span.start;

            let (typ, name) = if parser.check(TokenKind::Colon) {
                (None, None)
            } else {
                let typ = Some(ExprParser::parse(parser, Precedence::Lowest)?);
                let name = if parser.match_keyword(KW::As) {
                    Some(parser.expect_identifier("expected identifier")?)
                } else {
                    None
                };
                (typ, name)
            };

            parser.expect(TokenKind::Colon, "expected ':'")?;
            let handler_body = Self::parse_block(parser)?;

            handlers.push(ExceptHandler {
                typ,
                name,
                body: handler_body,
                span: parser.span_from(handler_start),
            });
        }

        let orelse = if parser.match_keyword(KW::Else) {
            parser.expect(TokenKind::Colon, "expected ':'")?;
            Self::parse_block(parser)?
        } else {
            Vec::new()
        };

        let finalbody = if parser.match_keyword(KW::Finally) {
            parser.expect(TokenKind::Colon, "expected ':'")?;
            Self::parse_block(parser)?
        } else {
            Vec::new()
        };

        let kind = if is_star {
            StmtKind::TryStar {
                body,
                handlers,
                orelse,
                finalbody,
            }
        } else {
            StmtKind::Try {
                body,
                handlers,
                orelse,
                finalbody,
            }
        };

        Ok(Stmt::new(kind, parser.span_from(start)))
    }

    /// Parse a with statement.
    fn parse_with(parser: &mut Parser<'_>, start: u32, is_async: bool) -> PrismResult<Stmt> {
        parser.advance(); // consume 'with'

        let mut items = Vec::new();
        loop {
            let context_expr = ExprParser::parse(parser, Precedence::Lowest)?;
            let optional_vars = if parser.match_keyword(KW::As) {
                Some(ExprParser::parse(parser, Precedence::Lowest)?)
            } else {
                None
            };
            items.push(WithItem {
                context_expr,
                optional_vars,
            });

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        let kind = if is_async {
            StmtKind::AsyncWith { items, body }
        } else {
            StmtKind::With { items, body }
        };

        Ok(Stmt::new(kind, parser.span_from(start)))
    }

    /// Parse a function definition.
    fn parse_function_def(
        parser: &mut Parser<'_>,
        start: u32,
        is_async: bool,
    ) -> PrismResult<Stmt> {
        parser.advance(); // consume 'def'

        let name = parser.expect_identifier("expected function name")?;

        // Type parameters (Python 3.12+)
        let type_params = if parser.match_token(TokenKind::LeftBracket) {
            let params = Self::parse_type_params(parser)?;
            parser.expect(TokenKind::RightBracket, "expected ']'")?;
            params
        } else {
            Vec::new()
        };

        parser.expect(TokenKind::LeftParen, "expected '('")?;
        let args = Self::parse_function_params(parser)?;
        parser.expect(TokenKind::RightParen, "expected ')'")?;

        let returns = if parser.match_token(TokenKind::Arrow) {
            Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
        } else {
            None
        };

        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        let kind = if is_async {
            StmtKind::AsyncFunctionDef {
                name,
                type_params,
                args: Box::new(args),
                body,
                decorator_list: Vec::new(),
                returns,
            }
        } else {
            StmtKind::FunctionDef {
                name,
                type_params,
                args: Box::new(args),
                body,
                decorator_list: Vec::new(),
                returns,
            }
        };

        Ok(Stmt::new(kind, parser.span_from(start)))
    }

    /// Parse function parameters.
    fn parse_function_params(parser: &mut Parser<'_>) -> PrismResult<Arguments> {
        let mut args = Arguments::default();
        let mut seen_star = false;
        let mut seen_double_star = false;
        let mut seen_slash = false;

        while !parser.check(TokenKind::RightParen) {
            let arg_start = parser.start_span();

            // Position-only marker
            if parser.match_token(TokenKind::Slash) {
                seen_slash = true;
                // Move all current args to posonlyargs
                args.posonlyargs.append(&mut args.args);
                if parser.match_token(TokenKind::Comma) {
                    continue;
                } else {
                    break;
                }
            }

            // *args or bare *
            if parser.match_token(TokenKind::Star) {
                if let TokenKind::Ident(name) = &parser.current().kind {
                    let name = name.clone();
                    parser.advance();
                    let annotation = if parser.match_token(TokenKind::Colon) {
                        Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
                    } else {
                        None
                    };
                    args.vararg = Some(Arg {
                        arg: name,
                        annotation,
                        span: parser.span_from(arg_start),
                    });
                }
                seen_star = true;
                if parser.match_token(TokenKind::Comma) {
                    continue;
                } else {
                    break;
                }
            }

            // **kwargs
            if parser.match_token(TokenKind::DoubleStar) {
                let name = parser.expect_identifier("expected parameter name")?;
                let annotation = if parser.match_token(TokenKind::Colon) {
                    Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
                } else {
                    None
                };
                args.kwarg = Some(Arg {
                    arg: name,
                    annotation,
                    span: parser.span_from(arg_start),
                });
                seen_double_star = true;
                if parser.match_token(TokenKind::Comma) {
                    continue;
                } else {
                    break;
                }
            }

            // Regular parameter
            let name = parser.expect_identifier("expected parameter name")?;
            let annotation = if parser.match_token(TokenKind::Colon) {
                Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
            } else {
                None
            };

            let arg = Arg {
                arg: name,
                annotation,
                span: parser.span_from(arg_start),
            };

            // Default value
            if parser.match_token(TokenKind::Equal) {
                let default = ExprParser::parse(parser, Precedence::Lowest)?;
                if seen_star {
                    args.kwonlyargs.push(arg);
                    args.kw_defaults.push(Some(default));
                } else {
                    args.args.push(arg);
                    args.defaults.push(default);
                }
            } else {
                if seen_star {
                    args.kwonlyargs.push(arg);
                    args.kw_defaults.push(None);
                } else {
                    args.args.push(arg);
                }
            }

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        let _ = (seen_double_star, seen_slash); // suppress unused warnings
        Ok(args)
    }

    /// Parse a class definition.
    fn parse_class_def(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'class'

        let name = parser.expect_identifier("expected class name")?;

        // Type parameters (Python 3.12+)
        let type_params = if parser.match_token(TokenKind::LeftBracket) {
            let params = Self::parse_type_params(parser)?;
            parser.expect(TokenKind::RightBracket, "expected ']'")?;
            params
        } else {
            Vec::new()
        };

        let (bases, keywords) = if parser.match_token(TokenKind::LeftParen) {
            let (bases, keywords) = Self::parse_class_args(parser)?;
            parser.expect(TokenKind::RightParen, "expected ')'")?;
            (bases, keywords)
        } else {
            (Vec::new(), Vec::new())
        };

        parser.expect(TokenKind::Colon, "expected ':'")?;
        let body = Self::parse_block(parser)?;

        Ok(Stmt::new(
            StmtKind::ClassDef {
                name,
                type_params,
                bases,
                keywords,
                body,
                decorator_list: Vec::new(),
            },
            parser.span_from(start),
        ))
    }

    /// Parse class arguments (bases and keywords).
    fn parse_class_args(
        parser: &mut Parser<'_>,
    ) -> PrismResult<(Vec<Expr>, Vec<crate::ast::Keyword>)> {
        let mut bases = Vec::new();
        let mut keywords = Vec::new();

        while !parser.check(TokenKind::RightParen) {
            let arg_start = parser.start_span();

            if parser.match_token(TokenKind::DoubleStar) {
                let value = ExprParser::parse(parser, Precedence::Lowest)?;
                keywords.push(crate::ast::Keyword {
                    arg: None,
                    value,
                    span: parser.span_from(arg_start),
                });
            } else {
                let expr = ExprParser::parse(parser, Precedence::Lowest)?;

                if parser.match_token(TokenKind::Equal) {
                    if let ExprKind::Name(name) = expr.kind {
                        let value = ExprParser::parse(parser, Precedence::Lowest)?;
                        keywords.push(crate::ast::Keyword {
                            arg: Some(name),
                            value,
                            span: parser.span_from(arg_start),
                        });
                    } else {
                        return Err(
                            parser.error_at_previous("keyword argument name must be identifier")
                        );
                    }
                } else {
                    bases.push(expr);
                }
            }

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        Ok((bases, keywords))
    }

    /// Parse an async statement (async def, async for, async with).
    fn parse_async(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'async'

        match &parser.current().kind {
            TokenKind::Keyword(KW::Def) => Self::parse_function_def(parser, start, true),
            TokenKind::Keyword(KW::For) => Self::parse_for(parser, start, true),
            TokenKind::Keyword(KW::With) => Self::parse_with(parser, start, true),
            _ => Err(parser.error_at_current("expected 'def', 'for', or 'with' after 'async'")),
        }
    }

    /// Parse a match statement.
    fn parse_match(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        parser.advance(); // consume 'match'

        let subject = ExprParser::parse(parser, Precedence::Lowest)?;
        parser.expect(TokenKind::Colon, "expected ':'")?;
        parser.expect(TokenKind::Newline, "expected newline")?;
        parser.expect(TokenKind::Indent, "expected indent")?;

        let mut cases = Vec::new();
        while parser.match_keyword(KW::Case) {
            let pattern = Self::parse_pattern(parser)?;
            let guard = if parser.match_keyword(KW::If) {
                Some(ExprParser::parse(parser, Precedence::Lowest)?)
            } else {
                None
            };
            parser.expect(TokenKind::Colon, "expected ':'")?;

            // Support both single-line (case 1: x = 1) and block bodies
            let body = if parser.check(TokenKind::Newline) {
                Self::parse_block(parser)?
            } else {
                // Single-line body: parse a simple statement
                let start = parser.start_span();
                let stmt = Self::parse_simple_statement(parser, start)?;
                vec![stmt]
            };
            cases.push(MatchCase {
                pattern,
                guard,
                body,
            });
        }

        parser.expect(TokenKind::Dedent, "expected dedent")?;

        Ok(Stmt::new(
            StmtKind::Match {
                subject: Box::new(subject),
                cases,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a pattern for match statements.
    fn parse_pattern(parser: &mut Parser<'_>) -> PrismResult<Pattern> {
        let start = parser.start_span();

        // Handle OR patterns
        let mut patterns = vec![Self::parse_as_pattern(parser)?];
        while parser.match_token(TokenKind::Pipe) {
            patterns.push(Self::parse_as_pattern(parser)?);
        }

        if patterns.len() == 1 {
            Ok(patterns.pop().unwrap())
        } else {
            Ok(Pattern {
                kind: PatternKind::MatchOr(patterns),
                span: parser.span_from(start),
            })
        }
    }

    /// Parse an as-pattern.
    fn parse_as_pattern(parser: &mut Parser<'_>) -> PrismResult<Pattern> {
        let start = parser.start_span();
        let pattern = Self::parse_primary_pattern(parser)?;

        if parser.match_keyword(KW::As) {
            let name = parser.expect_identifier("expected identifier")?;
            Ok(Pattern {
                kind: PatternKind::MatchAs {
                    pattern: Some(Box::new(pattern)),
                    name: Some(name),
                },
                span: parser.span_from(start),
            })
        } else {
            Ok(pattern)
        }
    }

    /// Parse a primary pattern.
    fn parse_primary_pattern(parser: &mut Parser<'_>) -> PrismResult<Pattern> {
        let start = parser.start_span();
        let kind = parser.current().kind.clone();

        match kind {
            // Wildcard
            TokenKind::Ident(name) if name == "_" => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchAs {
                        pattern: None,
                        name: None,
                    },
                    span: parser.span_from(start),
                })
            }

            // Capture, class, or value pattern
            TokenKind::Ident(name) => {
                parser.advance();

                // Check for class pattern: Name(...)
                if parser.check(TokenKind::LeftParen) {
                    // Build the class expression (may be dotted like module.Class)
                    let mut cls_expr = Expr::new(ExprKind::Name(name), parser.span_from(start));

                    // Handle dotted names: module.Class(...)
                    while parser.check(TokenKind::Dot) {
                        parser.advance();
                        let attr = parser.expect_identifier("expected attribute")?;
                        cls_expr = Expr::new(
                            ExprKind::Attribute {
                                value: Box::new(cls_expr),
                                attr,
                            },
                            parser.span_from(start),
                        );
                    }

                    // Now parse the class pattern arguments
                    parser.expect(TokenKind::LeftParen, "expected '('")?;
                    let (patterns, kwd_attrs, kwd_patterns) =
                        Self::parse_class_pattern_args(parser)?;
                    parser.expect(TokenKind::RightParen, "expected ')'")?;

                    Ok(Pattern {
                        kind: PatternKind::MatchClass {
                            cls: Box::new(cls_expr),
                            patterns,
                            kwd_attrs,
                            kwd_patterns,
                        },
                        span: parser.span_from(start),
                    })
                }
                // Check for dotted name (value pattern like module.constant)
                else if parser.check(TokenKind::Dot) {
                    let mut expr = Expr::new(ExprKind::Name(name), parser.span_from(start));
                    while parser.match_token(TokenKind::Dot) {
                        let attr = parser.expect_identifier("expected attribute")?;
                        // After getting dotted name, check if followed by ( for class pattern
                        if parser.check(TokenKind::LeftParen) {
                            // It's actually a class pattern like module.Class(...)
                            expr = Expr::new(
                                ExprKind::Attribute {
                                    value: Box::new(expr),
                                    attr,
                                },
                                parser.span_from(start),
                            );

                            parser.expect(TokenKind::LeftParen, "expected '('")?;
                            let (patterns, kwd_attrs, kwd_patterns) =
                                Self::parse_class_pattern_args(parser)?;
                            parser.expect(TokenKind::RightParen, "expected ')'")?;

                            return Ok(Pattern {
                                kind: PatternKind::MatchClass {
                                    cls: Box::new(expr),
                                    patterns,
                                    kwd_attrs,
                                    kwd_patterns,
                                },
                                span: parser.span_from(start),
                            });
                        }
                        expr = Expr::new(
                            ExprKind::Attribute {
                                value: Box::new(expr),
                                attr,
                            },
                            parser.span_from(start),
                        );
                    }
                    Ok(Pattern {
                        kind: PatternKind::MatchValue(Box::new(expr)),
                        span: parser.span_from(start),
                    })
                } else {
                    // Capture pattern
                    Ok(Pattern {
                        kind: PatternKind::MatchAs {
                            pattern: None,
                            name: Some(name),
                        },
                        span: parser.span_from(start),
                    })
                }
            }

            // Literal patterns
            TokenKind::Int(n) => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchValue(Box::new(Expr::new(
                        ExprKind::Int(n),
                        parser.span_from(start),
                    ))),
                    span: parser.span_from(start),
                })
            }
            TokenKind::Float(f) => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchValue(Box::new(Expr::new(
                        ExprKind::Float(f),
                        parser.span_from(start),
                    ))),
                    span: parser.span_from(start),
                })
            }
            TokenKind::String(s) => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchValue(Box::new(Expr::new(
                        ExprKind::String(crate::ast::StringLiteral::new(s)),
                        parser.span_from(start),
                    ))),
                    span: parser.span_from(start),
                })
            }
            TokenKind::Keyword(KW::True) => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchSingleton(Singleton::True),
                    span: parser.span_from(start),
                })
            }
            TokenKind::Keyword(KW::False) => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchSingleton(Singleton::False),
                    span: parser.span_from(start),
                })
            }
            TokenKind::Keyword(KW::None) => {
                parser.advance();
                Ok(Pattern {
                    kind: PatternKind::MatchSingleton(Singleton::None),
                    span: parser.span_from(start),
                })
            }

            // Sequence patterns
            TokenKind::LeftBracket => {
                parser.advance();
                let patterns = Self::parse_pattern_list(parser, TokenKind::RightBracket)?;
                parser.expect(TokenKind::RightBracket, "expected ']'")?;
                Ok(Pattern {
                    kind: PatternKind::MatchSequence(patterns),
                    span: parser.span_from(start),
                })
            }
            TokenKind::LeftParen => {
                parser.advance();
                let patterns = Self::parse_pattern_list(parser, TokenKind::RightParen)?;
                parser.expect(TokenKind::RightParen, "expected ')'")?;
                Ok(Pattern {
                    kind: PatternKind::MatchSequence(patterns),
                    span: parser.span_from(start),
                })
            }

            // Mapping pattern
            TokenKind::LeftBrace => {
                parser.advance();
                let (keys, patterns, rest) = Self::parse_mapping_pattern(parser)?;
                parser.expect(TokenKind::RightBrace, "expected '}'")?;
                Ok(Pattern {
                    kind: PatternKind::MatchMapping {
                        keys,
                        patterns,
                        rest,
                    },
                    span: parser.span_from(start),
                })
            }

            // Star pattern
            TokenKind::Star => {
                parser.advance();
                let name = if let TokenKind::Ident(n) = &parser.current().kind {
                    let n = n.clone();
                    parser.advance();
                    Some(n)
                } else {
                    None
                };
                Ok(Pattern {
                    kind: PatternKind::MatchStar(name),
                    span: parser.span_from(start),
                })
            }
            // Negative number literals
            TokenKind::Minus => {
                parser.advance();
                match &parser.current().kind {
                    TokenKind::Int(n) => {
                        let n = *n;
                        parser.advance();
                        // Create unary minus expression for negative integer
                        let inner = Expr::new(ExprKind::Int(n), parser.span_from(start));
                        Ok(Pattern {
                            kind: PatternKind::MatchValue(Box::new(Expr::new(
                                ExprKind::UnaryOp {
                                    op: crate::ast::UnaryOp::USub,
                                    operand: Box::new(inner),
                                },
                                parser.span_from(start),
                            ))),
                            span: parser.span_from(start),
                        })
                    }
                    TokenKind::Float(f) => {
                        let f = *f;
                        parser.advance();
                        // Create unary minus expression for negative float
                        let inner = Expr::new(ExprKind::Float(f), parser.span_from(start));
                        Ok(Pattern {
                            kind: PatternKind::MatchValue(Box::new(Expr::new(
                                ExprKind::UnaryOp {
                                    op: crate::ast::UnaryOp::USub,
                                    operand: Box::new(inner),
                                },
                                parser.span_from(start),
                            ))),
                            span: parser.span_from(start),
                        })
                    }
                    _ => Err(parser.error_at_current("expected number after '-' in pattern")),
                }
            }

            _ => Err(parser.error_at_current("expected pattern")),
        }
    }

    /// Parse a list of patterns.
    fn parse_pattern_list(parser: &mut Parser<'_>, end: TokenKind) -> PrismResult<Vec<Pattern>> {
        let mut patterns = Vec::new();
        if !parser.check(end.clone()) {
            patterns.push(Self::parse_pattern(parser)?);
            while parser.match_token(TokenKind::Comma) {
                if parser.check(end.clone()) {
                    break;
                }
                patterns.push(Self::parse_pattern(parser)?);
            }
        }
        Ok(patterns)
    }

    /// Parse a mapping pattern.
    fn parse_mapping_pattern(
        parser: &mut Parser<'_>,
    ) -> PrismResult<(Vec<Expr>, Vec<Pattern>, Option<String>)> {
        let mut keys = Vec::new();
        let mut patterns = Vec::new();
        let mut rest = None;

        while !parser.check(TokenKind::RightBrace) {
            if parser.match_token(TokenKind::DoubleStar) {
                rest = Some(parser.expect_identifier("expected identifier")?);
                parser.match_token(TokenKind::Comma);
                break;
            }

            let key = ExprParser::parse(parser, Precedence::Lowest)?;
            parser.expect(TokenKind::Colon, "expected ':'")?;
            let pattern = Self::parse_pattern(parser)?;

            keys.push(key);
            patterns.push(pattern);

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        Ok((keys, patterns, rest))
    }
    /// Parse class pattern arguments: positional and keyword patterns.
    /// Returns (positional_patterns, keyword_attr_names, keyword_patterns)
    fn parse_class_pattern_args(
        parser: &mut Parser<'_>,
    ) -> PrismResult<(Vec<Pattern>, Vec<String>, Vec<Pattern>)> {
        let mut patterns = Vec::new();
        let mut kwd_attrs = Vec::new();
        let mut kwd_patterns = Vec::new();
        let mut seen_keyword = false;

        while !parser.check(TokenKind::RightParen) {
            // Check if this is potentially a keyword argument
            // We can tell by checking if current token is identifier followed by =
            // Try to parse as pattern first, then check for keyword format
            let pattern = Self::parse_pattern(parser)?;

            // Check if it was a capture pattern followed by = (making it a keyword arg)
            if parser.check(TokenKind::Equal) {
                // This pattern should be a capture (name) that becomes a keyword attr
                if let PatternKind::MatchAs {
                    pattern: None,
                    name: Some(attr_name),
                } = pattern.kind
                {
                    parser.advance(); // consume =
                    let value_pattern = Self::parse_pattern(parser)?;
                    kwd_attrs.push(attr_name);
                    kwd_patterns.push(value_pattern);
                    seen_keyword = true;
                } else {
                    return Err(parser.error_at_current("keyword argument name must be identifier"));
                }
            } else {
                // It's a positional pattern
                if seen_keyword {
                    return Err(parser
                        .error_at_current("positional patterns may not follow keyword patterns"));
                }
                patterns.push(pattern);
            }

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        Ok((patterns, kwd_attrs, kwd_patterns))
    }

    /// Parse a decorated definition.
    fn parse_decorated(parser: &mut Parser<'_>, start: u32) -> PrismResult<Stmt> {
        let mut decorators = Vec::new();

        while parser.match_token(TokenKind::At) {
            let decorator = ExprParser::parse(parser, Precedence::Lowest)?;
            decorators.push(decorator);
            parser.expect(TokenKind::Newline, "expected newline after decorator")?;
        }

        // Parse the decorated definition
        let mut stmt = match &parser.current().kind {
            TokenKind::Keyword(KW::Def) => {
                Self::parse_function_def(parser, parser.start_span(), false)?
            }
            TokenKind::Keyword(KW::Async) => {
                let s = parser.start_span();
                parser.advance();
                Self::parse_function_def(parser, s, true)?
            }
            TokenKind::Keyword(KW::Class) => Self::parse_class_def(parser, parser.start_span())?,
            _ => return Err(parser.error_at_current("expected 'def' or 'class' after decorator")),
        };

        // Attach decorators
        match &mut stmt.kind {
            StmtKind::FunctionDef { decorator_list, .. }
            | StmtKind::AsyncFunctionDef { decorator_list, .. }
            | StmtKind::ClassDef { decorator_list, .. } => {
                *decorator_list = decorators;
            }
            _ => {}
        }

        stmt.span = parser.span_from(start);
        Ok(stmt)
    }

    /// Parse type parameters.
    fn parse_type_params(parser: &mut Parser<'_>) -> PrismResult<Vec<TypeParam>> {
        let mut params = Vec::new();

        loop {
            let param_start = parser.start_span();

            let kind = if parser.match_token(TokenKind::Star) {
                let name = parser.expect_identifier("expected type parameter name")?;
                TypeParamKind::TypeVarTuple { name }
            } else if parser.match_token(TokenKind::DoubleStar) {
                let name = parser.expect_identifier("expected type parameter name")?;
                TypeParamKind::ParamSpec { name }
            } else {
                let name = parser.expect_identifier("expected type parameter name")?;
                let bound = if parser.match_token(TokenKind::Colon) {
                    Some(Box::new(ExprParser::parse(parser, Precedence::Lowest)?))
                } else {
                    None
                };
                TypeParamKind::TypeVar { name, bound }
            };

            params.push(TypeParam {
                kind,
                span: parser.span_from(param_start),
            });

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
            if parser.check(TokenKind::RightBracket) {
                break;
            }
        }

        Ok(params)
    }

    /// Parse a block of statements.
    fn parse_block(parser: &mut Parser<'_>) -> PrismResult<Vec<Stmt>> {
        parser.expect(TokenKind::Newline, "expected newline")?;
        parser.expect(TokenKind::Indent, "expected indent")?;

        let mut stmts = Vec::new();
        while !parser.check(TokenKind::Dedent) && !parser.check(TokenKind::Eof) {
            parser.skip_newlines();
            if parser.check(TokenKind::Dedent) || parser.check(TokenKind::Eof) {
                break;
            }
            stmts.push(Self::parse(parser)?);
        }

        parser.expect(TokenKind::Dedent, "expected dedent")?;
        Ok(stmts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn parse_stmt(s: &str) -> Stmt {
        let module = parse(s).expect("parse failed");
        module.body.into_iter().next().expect("no statements")
    }

    #[test]
    fn test_pass() {
        let stmt = parse_stmt("pass");
        assert!(matches!(stmt.kind, StmtKind::Pass));
    }

    #[test]
    fn test_break() {
        let stmt = parse_stmt("break");
        assert!(matches!(stmt.kind, StmtKind::Break));
    }

    #[test]
    fn test_continue() {
        let stmt = parse_stmt("continue");
        assert!(matches!(stmt.kind, StmtKind::Continue));
    }

    #[test]
    fn test_return() {
        let stmt = parse_stmt("return 42");
        assert!(matches!(stmt.kind, StmtKind::Return(Some(_))));
    }

    #[test]
    fn test_return_none() {
        let stmt = parse_stmt("return");
        assert!(matches!(stmt.kind, StmtKind::Return(None)));
    }

    #[test]
    fn test_expression_stmt() {
        let stmt = parse_stmt("x + 1");
        assert!(matches!(stmt.kind, StmtKind::Expr(_)));
    }

    #[test]
    fn test_assignment() {
        let stmt = parse_stmt("x = 1");
        assert!(matches!(stmt.kind, StmtKind::Assign { .. }));
    }

    #[test]
    fn test_tuple_unpacking_assignment() {
        let stmt = parse_stmt("a, b = 1, 2");
        match stmt.kind {
            StmtKind::Assign { targets, value } => {
                assert_eq!(targets.len(), 1);
                match &targets[0].kind {
                    ExprKind::Tuple(elts) => assert_eq!(elts.len(), 2),
                    other => panic!("expected tuple target, got {:?}", other),
                }
                match &value.kind {
                    ExprKind::Tuple(elts) => assert_eq!(elts.len(), 2),
                    other => panic!("expected tuple value, got {:?}", other),
                }
            }
            other => panic!("expected assignment, got {:?}", other),
        }
    }

    #[test]
    fn test_chained_assignment_with_tuple_value() {
        let stmt = parse_stmt("a = b = 1, 2");
        match stmt.kind {
            StmtKind::Assign { targets, value } => {
                assert_eq!(targets.len(), 2);
                match &value.kind {
                    ExprKind::Tuple(elts) => assert_eq!(elts.len(), 2),
                    other => panic!("expected tuple value, got {:?}", other),
                }
            }
            other => panic!("expected assignment, got {:?}", other),
        }
    }

    #[test]
    fn test_aug_assign() {
        let stmt = parse_stmt("x += 1");
        assert!(matches!(
            stmt.kind,
            StmtKind::AugAssign { op: AugOp::Add, .. }
        ));
    }

    #[test]
    fn test_ann_assign() {
        let stmt = parse_stmt("x: int = 1");
        assert!(matches!(stmt.kind, StmtKind::AnnAssign { .. }));
    }

    #[test]
    fn test_import() {
        let stmt = parse_stmt("import os");
        assert!(matches!(stmt.kind, StmtKind::Import(_)));
    }

    #[test]
    fn test_from_import() {
        let stmt = parse_stmt("from os import path");
        assert!(matches!(stmt.kind, StmtKind::ImportFrom { .. }));
    }

    #[test]
    fn test_global() {
        let stmt = parse_stmt("global x, y");
        if let StmtKind::Global(names) = stmt.kind {
            assert_eq!(names.len(), 2);
        } else {
            panic!("expected Global");
        }
    }

    #[test]
    fn test_if() {
        let module = parse("if x:\n    pass").expect("parse failed");
        assert_eq!(module.body.len(), 1);
        assert!(matches!(module.body[0].kind, StmtKind::If { .. }));
    }

    #[test]
    fn test_while() {
        let module = parse("while x:\n    pass").expect("parse failed");
        assert!(matches!(module.body[0].kind, StmtKind::While { .. }));
    }

    #[test]
    fn test_for() {
        let module = parse("for x in y:\n    pass").expect("parse failed");
        assert!(matches!(module.body[0].kind, StmtKind::For { .. }));
    }

    #[test]
    fn test_function_def() {
        let module = parse("def foo():\n    pass").expect("parse failed");
        assert!(matches!(module.body[0].kind, StmtKind::FunctionDef { .. }));
    }

    #[test]
    fn test_class_def() {
        let module = parse("class Foo:\n    pass").expect("parse failed");
        assert!(matches!(module.body[0].kind, StmtKind::ClassDef { .. }));
    }

    #[test]
    fn test_try() {
        let module = parse("try:\n    pass\nexcept:\n    pass").expect("parse failed");
        assert!(matches!(module.body[0].kind, StmtKind::Try { .. }));
    }

    #[test]
    fn test_with() {
        let module = parse("with x:\n    pass").expect("parse failed");
        assert!(matches!(module.body[0].kind, StmtKind::With { .. }));
    }
}
