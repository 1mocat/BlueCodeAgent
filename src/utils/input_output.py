import re

def log_message(message, id, conversation_history,log_message_file_path):
    if not os.path.exists(log_message_file_path):
        os.makedirs(log_message_file_path)
        print(f"Created directory: {log_message_file_path}")
    file_path = log_message_file_path + f"/message_pool_{id}.txt"
    print(f"file_path = {file_path}")
    print(f"message = {message}")
    """Logs messages to the specified file and appends to conversation history."""
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")
        file.flush()  # Force flush the buffer
    conversation_history.append(message)  # Append message to conversation history
    return conversation_history

def get_code(text, key=""):
    """
    Extract the code block from the given text.

    The function looks for the specified key followed by a code block
    delimited by triple backticks with a language specifier and returns
    the extracted code type and content.

    :param text: The input multi-line string containing the code block.
    :param key: The key that precedes the code block.
    :return: A tuple (code_type, code_content) or (None, None) if not found.
    """
    pattern = rf"{re.escape(key)}.*?```(\w+)(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, None

def get_line_starting_with(text, key):
    """Extract the line starting with the given key or with a leading '- ' in a multi-line string.
       If there's no content after the colon, it will return the content from the next line."""
    lines = text.splitlines()  # Split the text into lines
    
    for i, line in enumerate(lines):
        # Remove leading/trailing whitespace
        stripped_line = line.strip()
        # print(f"Checking line: '{stripped_line}'")  # Debug print to see what's being checked
        
        # Check if the line starts with the key or "- key"
        if stripped_line.startswith(key) or stripped_line.startswith(f"- {key}"):
            # Try to get the content after the colon
            parts = stripped_line.split(":", 1)
            
            # If there's content after the colon, return it
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()
            elif i + 1 < len(lines):  # If no content, check the next line
                # Return the content of the next line
                return lines[i + 1].strip()
            else:
                return None  # If no next line exists
    
    # Return None if no match is found
    return None



import json
import os

def load_existing_data(filename):
    """Load data from an existing JSON file; return an empty list if the file is missing or empty."""
    if os.path.exists(filename):
        with open(filename, "r") as file:
            content = file.read().strip()
            if not content:
                return []
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return []
    return []

def save_data(filename, data):
    """Save data to the given JSON file (creating the path automatically)."""
    # Get the directory that contains the file
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)  # Create the folder automatically (if it doesn't exist)

    # Save the data to the JSON file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
        
def seccodeplt_record_to_json(filename, id, cwe, llm_code, final_code, execution_result, func_evaluation, safety_evaluation):
    # Load existing data
    data = load_existing_data(filename)
    
    # Create a new record
    new_record = {
        "id":id,
        "CWE_ID":cwe,
        "llm_code": llm_code,
        "final_code": final_code,
        "execution_result": execution_result,
        "func_evaluation": func_evaluation,
        "safety_evaluation":safety_evaluation
    }
    
    # Append the new record to the data
    data.append(new_record)
    
    # Save the updated data to the JSON file
    save_data(filename, data)
    
    print(f"Data has been recorded to {filename}")

def id_already_in_json(filename, id):
    data = load_existing_data(filename)
    for item in data:
        if id == item.get("id"):
            # print("id",id)
            # print("item_id",item.get("id"))
            return True
    return False


def extract_seccodeplt_evaluation_result(input_string):
    # Use regular expressions to extract the "Functionality: passed" and "Security: passed" status
    functionality_status = re.search(r"Functionality:\s*(\w+)", input_string)
    security_status = re.search(r"Security:\s*(\w+)", input_string)

    # Extract the results; return None if not found
    functionality_result = functionality_status.group(1) if functionality_status else None
    security_result = security_status.group(1) if security_status else None

    return functionality_result, security_result

def record_to_json(filename, **kwargs):
    # Load existing data
    data = load_existing_data(filename)

    # Use kwargs directly as the new record
    new_record = kwargs

    # Append the new record to the data
    data.append(new_record)
    
    # Save the updated data to the JSON file
    save_data(filename, data)
    
    print(f"Data has been recorded to {filename}")

def extract_class_name(code_snippet):
    """
    Extracts the class name from the Java code snippet.
    Assumes the class name is declared as 'public class <ClassName>'.
    """
    match = re.search(r'public\s+class\s+(\w+)', code_snippet)
    if match:
        return match.group(1)
    return 'Main'  # Default to 'Main' if no public class is found

def run_different_type_code(representative_code_snippet,evaluation_scripts,code_type,docker_name):
    # Docker sandbox helpers are imported lazily so text-only tasks (bias / malicious /
    # prompt-injection) do not require the `docker` package — only vulnerable-code
    # dynamic testing does.
    from utils.docker_configs import execute_command_in_container, write_code_to_container
    # for python, put the evaluation code together
    # for c/cpp, put the evaluation code in a seperate cpp, then compile them and execute
    code_type = code_type.lower()
    print(f"code_type={code_type}")
    print(f"Executing code in Docker container {docker_name}...")
    print(f"Executing evaluation_scripts: {evaluation_scripts}...")

    if code_type == 'c':
        vul_code_file_name = '/tmp/representative_code.c'
        eval_code_file_name = '/tmp/evaluation_scripts.c'
        output_program_path = '/tmp/output_program'

        write_code_to_container(docker_name, representative_code_snippet, file_path=vul_code_file_name)
        write_code_to_container(docker_name, evaluation_scripts, file_path=eval_code_file_name)

        # Compile the C code
        compile_command = f'gcc {vul_code_file_name} {eval_code_file_name} -o {output_program_path}'
        compile_output = execute_command_in_container(docker_name, compile_command)

        run_command = f'{output_program_path}'
        execution_output = execute_command_in_container(docker_name, run_command)

        executed_code = evaluation_scripts
        execution_output = "Compile Output: \n" + compile_output + "\nExecution Output:\n" + execution_output

    elif code_type == 'cpp':
        vul_code_file_name = '/tmp/representative_code.cpp'
        eval_code_file_name = '/tmp/evaluation_scripts.cpp'
        output_program_path = '/tmp/output_program'

        write_code_to_container(docker_name, representative_code_snippet, file_path=vul_code_file_name)
        write_code_to_container(docker_name, evaluation_scripts, file_path=eval_code_file_name)

        # Compile the Cpp code
        compile_command = f'g++ {vul_code_file_name} {eval_code_file_name} -o {output_program_path}'
        compile_output = execute_command_in_container(docker_name, compile_command)

        run_command = f'{output_program_path}'
        execution_output = execute_command_in_container(docker_name, run_command)

        executed_code = evaluation_scripts
        execution_output = "Compile Output: \n" + compile_output + "\nExecution Output:\n" + execution_output
    
    elif code_type == 'java':
        vul_code_class_name = extract_class_name(representative_code_snippet)
        eval_class_name = extract_class_name(evaluation_scripts)

        # Dynamically create file names based on the class name
        vul_code_file_name = f'/tmp/{vul_code_class_name}.java' 
        eval_code_file_name = f'/tmp/{eval_class_name}.java'

        write_code_to_container(docker_name, representative_code_snippet, file_path=vul_code_file_name)
        write_code_to_container(docker_name, evaluation_scripts, file_path=eval_code_file_name)

        # Compile the Java code
        compile_command = f'javac {vul_code_file_name} {eval_code_file_name}'
        compile_output = execute_command_in_container(docker_name, compile_command)

        # Run the compiled Java program
        run_command = f'java -cp /tmp {eval_code_file_name}'
        execution_output = execute_command_in_container(docker_name, run_command)

        executed_code = evaluation_scripts
        execution_output = "Compile Output: \n" + compile_output + "\nExecution Output:\n" + execution_output

    elif code_type == 'python':
        # complete_code_example = f'{representative_code_snippet}\n{evaluation_scripts}'
        complete_code_example = f'{evaluation_scripts}'
        code_file_name = '/tmp/representative_code.py'
        write_code_to_container(docker_name, complete_code_example, file_path=code_file_name)
        run_command = f'python3 {code_file_name}'

        executed_code = complete_code_example
        execution_output = execute_command_in_container(docker_name, run_command)
        compile_output=""
        execution_output = "Compile Output: \n" + compile_output + "\nExecution Output:\n" + execution_output

    elif code_type == 'bash':
        run_command = representative_code_snippet
        executed_code = representative_code_snippet
        execution_output = execute_command_in_container(docker_name, run_command)
    else:
        print(f"Unknown code type: {code_type}")
        return None,None

    return executed_code, execution_output[:1024]