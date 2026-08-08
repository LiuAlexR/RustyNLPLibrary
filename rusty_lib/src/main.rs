// use rusty_lib::components::*;
use rusty_lib::util;
fn main() { 
    let x = util::retrieve_source("liu_hello_world.txt");
    println!("{x}");
    let _ = util::write_to_storage("test.txt", &"boo\nbooo".to_string());
}
