

/// Passing test
#[test]
fn passing_test() -> () {
    assert_eq!(1, 1);
}
/// Failing test
#[ignore = "failing on purpose"]
#[test]
fn failing_test() -> () {
    assert_eq!(1, 2);
}
/// Reads a basic file. 
#[test]
fn read_basic() -> () {
    use crate::util;
    let content: String = util::retrieve_source("liu_basic.file");
    assert_eq!(content, "hello world".to_string());
}