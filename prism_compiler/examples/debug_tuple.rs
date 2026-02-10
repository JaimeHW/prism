// Minimal test to debug tuple unpacking from await expression

use prism_compiler::Compiler;
use prism_parser::Lexer;
use prism_parser::parse;

fn main() {
    let source = r#"
async def get_pair():
    return (1, 2)

async def main():
    a, b = await get_pair()
    return a + b
"#;

    // Debug: print all tokens first
    println!("=== Tokens ===");
    let mut lexer = Lexer::new(source);
    let mut count = 0;
    loop {
        let tok = lexer.next_token();
        println!("{:3}: {:?}", count, tok.kind);
        count += 1;
        if matches!(tok.kind, prism_parser::TokenKind::Eof) {
            break;
        }
        if count > 200 {
            println!("Token limit reached, possible lexer loop");
            break;
        }
    }

    println!("\n=== Parsing ===");
    let module = parse(source).expect("parse error");
    println!("Parsed successfully!");

    println!("Compiling...");
    let result = Compiler::compile_module(&module, "<test>");
    println!("Compile result: {:?}", result.is_ok());
}
