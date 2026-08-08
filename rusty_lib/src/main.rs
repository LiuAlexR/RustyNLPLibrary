// use rusty_lib::components::*;
use rusty_lib::util;
fn main() { 
    let x = util::retrieve_source("orwell_1984.txt");
    println!("{x}");
}
