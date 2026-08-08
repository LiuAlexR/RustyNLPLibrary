use crate::components::tokenizer::bpe_tokenize;

pub fn print_test() -> () {
    let x = "Kareem Abdul-Jabbar was the Finals MVP of the 1971 NBA FINALS, winning against the Los Angeles Lakers.";

    let v = bpe_tokenize(x, 200);
    println!("{:?}", v);
}
