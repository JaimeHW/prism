//! Debug test for bytecode inspection

use prism_compiler::*;
use prism_parser::parse;

fn main() {
    let source = r#"
x = 0
while x < 5:
    x = x + 1
"#;
    let module = parse(source).unwrap();
    let code = Compiler::compile_module(&module, "<test>").unwrap();
    println!("{}", disassemble(&code));
}
