use std::error::Error;

// Converts Vec<String> to Vec<CString>.as_ptr()
#[macro_export(local_inner_macros)]
macro_rules! cstring_vec {
    ($vec:ident) => { $vec.iter().map(|s: &String| CString::new(s.as_str()).unwrap()).collect::<Vec<_>>() };
}

// converts Vec<String> to Vec<&str>
#[macro_export(local_inner_macros)]
macro_rules! str_vec {
    ($vec:ident) => { $vec.iter().map(|s| s.as_str()).collect::<Vec<&str>>() };
}

// Properly process dependencies
pub fn legacy_process(src: &str, name: &str) -> String {
    let mut file_string = format!("\n// FILE: {}", name);
    for line in src.lines(){
        if !line.trim().starts_with("#include") {
            //println!("Legacy:  {}", line);
            file_string.push_str(line);
            file_string.push('\n');
        }
    }
    file_string
}
