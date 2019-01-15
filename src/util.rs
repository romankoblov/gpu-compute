use std::error::Error;

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
