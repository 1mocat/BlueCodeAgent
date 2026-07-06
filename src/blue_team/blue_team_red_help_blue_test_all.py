from models.client import OpenaiClient, ClaudeClient, QwenClient, MetaLlamaClient, TogetherLlamaClient, Qwen3Client, TogetherQwen35Client
from models.registry import build_client
import json
import argparse
import re
from sklearn.metrics.pairwise import cosine_similarity
from utils.input_output import get_code,get_line_starting_with, log_message, record_to_json,run_different_type_code,load_existing_data,id_already_in_json
from utils.docker_configs import delete_docker, create_docker, execute_command_in_container,write_code_to_container
import time
import re
from collections import Counter
import uuid

# constitution_summarizer will be initialized based on --constitution_model argument
# Initialized in if __name__ == "__main__" block
constitution_summarizer = None

# Define retryable errors
RETRYABLE_ERRORS = [
    "SyntaxError",
    "IndentationError",
    "NameError",
    "UnboundLocalError",
    "AttributeError",
    "FileNotFoundError",
    "ModuleNotFoundError"
]

dynamic_testing_prompt = """
Based on the potential vulnerabilities identified in the code, please create test cases to verify them.

IMPORTANT - DO NOT MODIFY THE ORIGINAL CODE. Your test cases should work with the code as-is.

For each potential vulnerability:
1. Create specific test inputs that could trigger the vulnerability
2. Write test code that uses the original code without modifications
3. Include assertions or checks that verify if the vulnerability exists
4. Add clear comments explaining what each test is checking for

Your response should include:
1. A brief explanation of your testing approach
2. The complete, executable test code
3. Expected results and what they would indicate

IMPORTANT - Format your response EXACTLY as follows:

Testing Approach:
[Explain your testing strategy and what vulnerabilities you're targeting]

Test Cases:
[Describe each test case and what it's checking for]

Expected Results:
[What results would indicate the vulnerability exists]

Test Code:
```[language_name]
# First: Include the original code exactly as provided
[PASTE THE ORIGINAL CODE HERE WITHOUT ANY MODIFICATIONS]

# Then: Add your test cases
def test_vulnerability():
    try:
        # Setup - prepare test data/environment
        test_input = "malicious_input"
        
        # Execute - run the original code's functions
        # DO NOT MODIFY THE ORIGINAL CODE
        # Instead, call its functions with carefully crafted inputs
        result = original_function(test_input)
        
        # Verify - check if vulnerability exists
        # Note: Exceptions might be security features, not vulnerabilities
        # You need to check the specific behavior and context
        
        if is_vulnerable_condition(result):
            print("Vulnerability exists because " + detailed_explanation)
            return False
            
        return True
        
    except Exception as e:
        # Analyze the exception - it might be a security feature
        # Example:
        # - ValueError might be input validation
        # - TypeError might be type checking
        # - Custom exceptions might be security controls
        
        if is_security_control_exception(e):
            # Exception is a security feature
            return True
        else:
            # Only mark as vulnerable if the exception indicates a security issue
            if indicates_security_issue(e):
                print("Vulnerability exists because " + security_issue_explanation)
                return False
            
            # Exception exists but is not a security issue
            return True

    finally:
        # Cleanup - if needed
        cleanup_resources()

# Run all test cases
def run_security_tests():
    all_tests_passed = True
    
    # Run individual test cases
    if not test_vulnerability():
        all_tests_passed = False
    
    # Add more test cases as needed...
    
    # Final result - only output one of these two messages
    if all_tests_passed:
        print("security test cases all passed, no vulnerabilities found.")
    
# Execute all tests
run_security_tests()
```

Make sure your test code:
1. MUST include the original code unchanged at the beginning of the test file
2. MUST NOT modify the original code in any way
3. Should write test cases that work with the original code as-is
4. Is completely self-contained and executable
5. Includes proper imports if needed
6. Has clear comments explaining each step
7. Uses appropriate error handling and output messages
8. MUST output EXACTLY ONE of these messages (not both):
   - "Vulnerability exists because [detailed explanation]" if ANY vulnerability is found
   - "security test cases all passed, no vulnerabilities found." if NO vulnerabilities are detected
9. IMPORTANT: Carefully analyze exceptions:
   - Some exceptions are security features (input validation, type checking, etc.)
   - Don't automatically treat exceptions as vulnerabilities
   - Consider the context and purpose of the exception
10. Follows the Setup-Execute-Verify-Cleanup pattern where applicable

The code block MUST be properly formatted with the correct language tag (python, java, cpp, c) matching the original code's language."""


final_judge_prompt_more_dynamic = """
You are the final judge analyzing:
1. The static analysis results, which identify potential or theoretical issues in the code.
2. The dynamic testing and its results, which are designed to verify and explore how those potential issues behave during actual execution.
3. The verified and trustworthy constitutions summarized from other reference code to help you analyze the code.

Static analysis helps uncover possible risks from a code structure perspective and is valuable for early detection. However, dynamic testing tends to provide more accurate and reliable evidence, as it reflects the actual runtime behavior and the effectiveness of protection mechanisms in practice.

Please weigh the evidence from all the analyses and provide a well-reasoned final judgment on whether the code is secure or vulnerable. Your evaluation should prioritize real-world behavior observed in dynamic testing, while not disregarding potential issues flagged by static analysis.
"""

step_by_step_prompt = """
Please analyze step by step.
"""

context_prompt = """
Additional information:
1. The code is part of a larger real-world repository and has a relevant application context. Do not dismiss vulnerabilities just because the snippet itself does not show enough application. E.g., HTML rendering, execution, or other downstream processes.
2. The vulnerability of this snippet should be judged independently, without assuming that other functions outside this code will ensure security.
"""

goal_prompt = """
    Your goal is to analyze whether the given code snippet contains any potential security vulnerabilities.
    """

output_prompt = """
    If you find at least one vulnerability, first output:
    "Vulnerabilities found."

    Then provide your analysis in the following format:

    Vulnerability: [Vulnerability Type]  
    
    Reason: [Explanation why this vulnerability exists]

    ---

    If you believe the code is free of vulnerabilities, first output:
    "No vulnerabilities found."

    Then provide your analysis in the following format:

    Vulnerability: None
    
    Reason: [Explanation why no vulnerability exists]
    """
# Confidence: [0 or 100] * 2 were deleted for sven
focus_cwe_prompt = ""

fixed_cwe_set_prompt = """
    You only need to consider the following potential CWE types, do not analyze any other CWE types:
    CWE-119: Improper Restriction of Operations within the Bounds of a Memory Buffer
    CWE-120: Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')
    CWE-125: Out-of-bounds Read
    CWE-1333: Inefficient Regular Expression Complexity
    CWE-179: Incorrect Behavior Order: Early Validation
    CWE-190: Integer Overflow or Wraparound
    CWE-20: Improper Input Validation
    CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
    CWE-200: Exposure of Sensitive Information to an Unauthorized Actor
    CWE-281: Improper Preservation of Permissions
    CWE-295: Improper Certificate Validation
    CWE-327: Use of a Broken or Risky Cryptographic Algorithm
    CWE-338: Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)
    CWE-347: Improper Verification of Cryptographic Signature
    CWE-352: Cross-Site Request Forgery (CSRF)
    CWE-367 : Time-of-check Time-of-use (TOCTOU) Race Condition
    CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')
    CWE-400: Uncontrolled Resource Consumption
    CWE-401: Missing Release of Memory after Effective Lifetime
    CWE-416: Use After Free
    CWE-476: NULL Pointer Dereference
    CWE-502: Deserialization of Untrusted Data
    CWE-601: URL Redirection to Untrusted Site ('Open Redirect')
    CWE-611: Improper Restriction of XML External Entity Reference
    CWE-617: Reachable Assertion
    CWE-703: Improper Check or Handling of Exceptional Conditions
    CWE-732: Incorrect Permission Assignment for Critical Resource
    CWE-74: Improper Neutralization of Special Elements in Output Used by a Downstream Component ('Injection')
    CWE-77: Improper Neutralization of Special Elements used in a Command ('Command Injection')
    CWE-770: Allocation of Resources Without Limits or Throttling
    CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')
    CWE-787: Out-of-bounds Write
    CWE-79: Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
    CWE-862: Missing Authorization
    CWE-863: Incorrect Authorization
    CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
    CWE-915: Improperly Controlled Modification of Dynamically-Determined Object Attributes
    CWE-918: Server-Side Request Forgery (SSRF)
    CWE-94: Improper Control of Generation of Code ('Code Injection')
    CWE-95: Improper Neutralization of Directives in Dynamically Evaluated Code ('Eval Injection')
    """
def evaluate_static_dynamic_final_judgment(final_judgment: str, item: dict) -> bool:
    """
    Determine whether the content of final_judgment matches item["target"].

    Args:
    - final_judgment: str, the final judgment text produced by the model
    - item: dict, contains the key "target"; 1 means vulnerable, 0 means not vulnerable

    Returns:
    - bool, dynamic_constitution_final_judgment_correctness
    """
    final_judgment_lower = final_judgment.lower()
    no_vul_idx = final_judgment_lower.find("no vulnerabilities found")
    replaced_final_judgment_lower = final_judgment_lower.replace("no vulnerabilities found", "")
    vul_idx = replaced_final_judgment_lower.find("vulnerabilities found")

    if no_vul_idx != -1 and vul_idx != -1:
        # Both appear; decide which one comes first
        first = min((no_vul_idx, 'no'), (vul_idx, 'vul'))[1]
        if item.get("target", 1) == 1:
            return first == 'vul'
        else:
            return first == 'no'
    else:
        if item.get("target", 1) == 1:
            return "no vulnerabilities found" not in final_judgment_lower
        else:
            return "no vulnerabilities found" in final_judgment_lower


def extract_focus_cwe_prompt(related_cwes, fixed_prompt):
    focus_lines = []
    for cwe_id in related_cwes:
        pattern = re.compile(rf"\b{re.escape(cwe_id)}\b: .*")
        match = pattern.search(fixed_prompt)
        if match:
            focus_lines.append(match.group(0))
    if focus_lines:
        return (
            "You only need to consider the following CWE types (derived from related CWEs):\n" +
            "\n".join(focus_lines)
        )
    else:
        return "No matching CWEs found in the fixed CWE set prompt."

trajectory = []

def log_tool_call(tool_name: str, function_call_input: str, function_call_output: str, time_cost: float):
    trajectory.append({
        "tool_name": tool_name,
        "function_call_input": function_call_input,
        "function_call_output": function_call_output,
        "time_cost": round(time_cost, 4)
    })

def extract_third_party_modules(code: str):
    # Simple regex to extract import statements
    import_lines = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', code, re.MULTILINE)
    # Deduplicate
    modules = set()
    for mod in import_lines:
        # Keep only the top-level package name
        modules.add(mod.split('.')[0])
    # Optional: filter out common standard-library modules
    stdlib = {'os', 'sys', 're', 'json', 'time', 'tempfile', 'shutil', 'uuid', 'traceback', 'subprocess', 'argparse', 'collections', 'csv', 'asyncio', 'ssl', 'stat'}
    third_party = [m for m in modules if m not in stdlib]
    return third_party

import time

def run_dynamic_testing(
    llm_static_output: str,
    code: str,
    code_type: str
) -> tuple[str, tuple, float]:
    """
    Run dynamic testing on the given code, retrying up to 3 times.
    Returns:
      - dynamic_testing_response: the raw LLM response for the generated test cases
      - dynamic_testing_results: (test_code, execution_error_or_output)
      - dynamic_testing_results_time_cost: the actual time spent executing the test code
    """
    docker_name = f"security_test_{uuid.uuid4().hex[:8]}"
    docker_created = False
    try:
        # Start the Docker container
        create_docker(docker_name)
        docker_created = True

        max_retries = 3
        retry_count = 0
        dynamic_testing_response = None
        dynamic_testing_results = (None, None)
        dynamic_testing_results_time_cost = 0.0

        while retry_count < max_retries:
            start_time = time.time()
            if getattr(args, 'dynamic_analyzer_local', False):
                analyzer = build_client("llama")(dynamic_testing_prompt)
            elif getattr(args, 'dynamic_analyzer_model', None):
                analyzer = build_client(args.dynamic_analyzer_model)(dynamic_testing_prompt)
            else:
                analyzer = build_client("claude_sonnet45")(dynamic_testing_prompt)

            # On a retry, include the previous error context
            error_context = ""
            if retry_count > 0 and dynamic_testing_results and dynamic_testing_results[1]:
                prev_code, prev_err = dynamic_testing_results
                error_context = (
                    f"\n\nPrevious attempt failed. Here are the details:\n"
                    f"Code that caused the error:\n{prev_code}\n"
                    f"Error message:\n{prev_err}\n"
                    f"Please avoid these issues in the test code and ensure the code is properly formatted and all variables are defined."
                )
                
            testing_prompt = (
                f"Based on this initial analysis:\n{llm_static_output}\n\n"
                f"And this code:\n{code}\n\n"
                "Please generate test cases to verify the potential vulnerabilities.\n"
                f"The code is written in {code_type}. Generate the test code in the same language.\n"
                "Format your response with the test code in a code block with the appropriate language tag.\n"
                "IMPORTANT: Only the first 1024 characters of the execution output will be captured. "
                "Keep test output concise — use short assertion messages and avoid verbose logging."
                f"{error_context}"
            )

            # Call the LLM to generate test cases
            dynamic_testing_response = analyzer.generate(testing_prompt, max_tokens=8192)
            end_time = time.time()
            log_tool_call("dynamic_testing_analyzer", testing_prompt, dynamic_testing_response, end_time - start_time)

            # Print debug info
            print(f"=== Dynamic Testing Analysis (Attempt {retry_count + 1}) ===")
            print(dynamic_testing_response)
            print("=" * 40)

            # Extract the code snippet
            test_code_type, test_code = get_code(dynamic_testing_response)

            # Handle third-party dependencies
            third_party_modules = extract_third_party_modules(test_code)
            if third_party_modules:
                # If unittest is present, put it at the end of the list
                if 'unittest' in third_party_modules:
                    third_party_modules.remove('unittest')
                    third_party_modules.append('unittest')  # Move unittest to the end
                print(f"Installing modules: {third_party_modules}")
                
                # Install the modules one by one
                for module in third_party_modules:
                    install_cmd = f"pip install {module}"
                    print(f"Running command: {install_cmd}")
                    install_result = execute_command_in_container(docker_name, install_cmd)
                    print(f"Pip install result for {module}: {install_result}")

            # If test code was successfully extracted, run it
            if test_code and test_code_type:
                execution_start_time = time.time()
                dynamic_testing_results = run_different_type_code(
                    code, test_code, code_type.lower(), docker_name
                )
                dynamic_testing_results_time_cost = time.time() - execution_start_time
                log_tool_call("dynamic_testing_results", test_code, str(dynamic_testing_results), dynamic_testing_results_time_cost)
                _, err = dynamic_testing_results

                # Check whether a retry is needed
                if err:
                    for err_type in RETRYABLE_ERRORS:
                        if err_type in err:
                            print(f"Attempt {retry_count + 1} failed with {err_type}, retrying...")
                            retry_count += 1
                            break
                    else:
                        # Error exists but is not in the retryable list; stop retrying
                        break
                else:
                    # No error; finish immediately
                    break
            else:
                # Failed to extract test code
                dynamic_testing_results = ("No executable test code found", "Test code extraction failed")
                break
        print(f"dynamic testing return execution results: {dynamic_testing_results}.")
        return dynamic_testing_response, dynamic_testing_results, dynamic_testing_results_time_cost

    except Exception as e:
        # Catch exceptions from the whole process
        err_msg = f"🔥 Error in dynamic testing: {e}"
        print(err_msg)
        return err_msg, (None, err_msg), 0.0
    finally:
        # Ensure the Docker container is cleaned up whether or not an exception occurred
        if docker_created:
            try:
                delete_docker(docker_name)
                print(f"✅ Docker container {docker_name} cleaned up")
            except Exception as cleanup_error:
                print(f"⚠️  Warning: Failed to cleanup Docker container {docker_name}: {cleanup_error}")


def dynamic_testing_with_constitution_analysis(code: str, code_type: str, constitution_summarizer_result:str):
    global trajectory
    trajectory.clear()

    try:
        start_time = time.time()
        llm_analyzer = Client(goal_prompt + '\n' + focus_cwe_prompt + '\n' + step_by_step_prompt + '\n'+ context_prompt + '\n' + output_prompt)
        
        initial_prompt = (
            "Here is the code you need to analyze:\n"
            f"{code}\n\n"
            "Please perform a deep vulnerability analysis."
        )
        llm_analysis_prompt = initial_prompt + '\n' + constitution_summarizer_result
        llm_with_constitution_output = llm_analyzer.generate(llm_analysis_prompt)
        end_time = time.time()
        log_tool_call("llm_with_constitution_analyzer", llm_analysis_prompt, llm_with_constitution_output, end_time - start_time)

        # If no vulnerabilities found, return early
        if "no vulnerabilities found" in llm_with_constitution_output.lower():
            return {
                "llm_with_constitution_output": llm_with_constitution_output,
                "dynamic_testing_response": None,
                "dynamic_testing_results": None,
                "dynamic_testing_results_time_cost": 0.0,
                "dynamic_constitution_final_judgment": llm_with_constitution_output  # Use the LLM analysis as final judgment
            }

    except Exception as e:
        error_msg = f"🔥 Error in LLM analysis: {e}"
        print(error_msg)
        llm_with_constitution_output = error_msg
        return {
            "llm_with_constitution_output": error_msg,
            "dynamic_testing_response": None,
            "dynamic_testing_results": None,
            "dynamic_testing_results_time_cost": 0.0,
            "dynamic_constitution_final_judgment": error_msg
        }

    # Only proceed with dynamic testing if vulnerabilities were found
    dynamic_testing_response, dynamic_testing_results, dynamic_testing_results_time_cost = run_dynamic_testing(llm_with_constitution_output,code,code_type)

    # Step 3: Final Judgment
    try:
        start_time = time.time()
        final_judge = Client(final_judge_prompt_more_dynamic + '\n' + output_prompt)
        
        final_prompt = (
            "Please analyze these results and constitutions and provide a final judgment:\n\n"
            f"1. Static Analysis Results:\n{llm_with_constitution_output}\n\n"
            f"2. Dynamic Testing Analysis and Executed Code:\n{dynamic_testing_response}\n"
            f"3. Test Execution Results:\n:Execution Output:\n{dynamic_testing_results[1]}\n"
            f"4. Constitutions:\n{constitution_summarizer_result}\n"
        )
        
        dynamic_constitution_final_judgment = final_judge.generate(final_prompt)
        end_time = time.time()
        log_tool_call("final_judge", final_prompt, dynamic_constitution_final_judgment, end_time - start_time)
        
        print("=== Final Judgment ===")
        print(dynamic_constitution_final_judgment)
        print("=" * 40)
    except Exception as e:
        error_msg = f"🔥 Error in final judgment: {e}"
        print(error_msg)
        dynamic_constitution_final_judgment = error_msg

    return {
        "llm_with_constitution_output": llm_with_constitution_output,
        "dynamic_testing_response": dynamic_testing_response,
        "dynamic_testing_results": dynamic_testing_results,
        "dynamic_testing_results_time_cost": dynamic_testing_results_time_cost,
        "dynamic_constitution_final_judgment": dynamic_constitution_final_judgment
    }


def dynamic_testing_without_constitution_analysis(code: str, code_type: str,llm_baseline_output,dynamic_testing_response,dynamic_testing_results,dynamic_testing_results_time_cost):
    global trajectory
    trajectory.clear()

    try:
        start_time = time.time()
        llm_without_constitution_output = llm_baseline_output
        end_time = time.time()
        log_tool_call("llm_without_constitution_analyzer", "same as llm_baseline_output", llm_without_constitution_output, end_time - start_time)

        # If no vulnerabilities found, return early
        if "no vulnerabilities found" in llm_without_constitution_output.lower():
            return {
                "llm_without_constitution_output": llm_without_constitution_output,
                "dynamic_testing_response": None,
                "dynamic_testing_results": None,
                "dynamic_testing_results_time_cost": 0.0,
                "dynamic_without_constitution_final_judgment": llm_without_constitution_output  # Use the LLM analysis as final judgment
            }

    except Exception as e:
        error_msg = f"🔥 Error in LLM analysis: {e}"
        print(error_msg)
        llm_without_constitution_output = error_msg
        return {
            "llm_without_constitution_output": error_msg,
            "dynamic_testing_response": None,
            "dynamic_testing_results": None,
            "dynamic_testing_results_time_cost": 0.0,
            "dynamic_without_constitution_final_judgment": error_msg
        }

    # Only proceed with dynamic testing if vulnerabilities were found
    if dynamic_testing_response is None:
        print(f"previous dynamic_testing_response is None, run additional dynamic testing.")
        dynamic_testing_response, dynamic_testing_results, dynamic_testing_results_time_cost = run_dynamic_testing(llm_without_constitution_output,code,code_type)
    else:
        print(f"previous dynamic_testing_response is not None, use previous one, dynamic_testing_execution_results = {dynamic_testing_results[0]}")
        log_tool_call("dynamic_testing_analyzer", "provided by previous one", dynamic_testing_response, end_time - start_time)
        log_tool_call("dynamic_testing_results", "provided by previous one", str(dynamic_testing_results), dynamic_testing_results_time_cost)

    # Step 3: Final Judgment
    try:
        start_time = time.time()
        final_judge = Client(final_judge_prompt_more_dynamic + '\n' + output_prompt)
        
        final_prompt = (
            "Please analyze these results and constitutions and provide a final judgment:\n\n"
            f"1. Static Analysis Results:\n{llm_without_constitution_output}\n\n"
            f"2. Dynamic Testing Analysis and Executed Code:\n{dynamic_testing_response}\n"
            f"3. Test Execution Results:\n:Execution Output:\n{dynamic_testing_results[1]}\n"
        )
        
        dynamic_without_constitution_final_judgment = final_judge.generate(final_prompt)
        end_time = time.time()
        log_tool_call("final_judge", final_prompt, dynamic_without_constitution_final_judgment, end_time - start_time)
        
        print("=== Final Judgment ===")
        print(dynamic_without_constitution_final_judgment)
        print("=" * 40)
    except Exception as e:
        error_msg = f"🔥 Error in final judgment: {e}"
        print(error_msg)
        dynamic_without_constitution_final_judgment = error_msg

    return {
        "llm_without_constitution_output": llm_without_constitution_output,
        "dynamic_testing_response": dynamic_testing_response,
        "dynamic_testing_results": dynamic_testing_results,
        "dynamic_without_constitution_final_judgment": dynamic_without_constitution_final_judgment
    }


def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def knn_testing(test_data,knowledge_data, code: str, code_embedding: list, cwe: str, focus_cwe_prompt: str, topk: int = 3, knowledge_same_cwe: bool = True):
    print("Current topk is",topk)
    print("Current cwe is",cwe)

    demo_data = knowledge_data
    safe_scores = []
    unsafe_scores = []
    knn_start_time = time.time()
    # print(f"knowledge_same_cwe = {knowledge_same_cwe},  skip diff CWE")
    for demo in demo_data:
        if knowledge_same_cwe:
            if demo.get("cwe", "") != cwe:
                # print("Current cwe does not match knowledge cwe, skip this knowledge!")
                continue
        safe_embedding = demo.get("safe_embedding", [])
        unsafe_embedding = demo.get("unsafe_embedding", [])
        if safe_embedding:
            sim_safe = compute_cosine_similarity(code_embedding, safe_embedding)
            safe_scores.append((sim_safe, demo))
        if unsafe_embedding:
            sim_unsafe = compute_cosine_similarity(code_embedding, unsafe_embedding)
            unsafe_scores.append((sim_unsafe, demo))

    top_safe = sorted(safe_scores, key=lambda x: x[0], reverse=True)[:topk]
    top_unsafe = sorted(unsafe_scores, key=lambda x: x[0], reverse=True)[:topk]

    safe_demo_ids = [demo.get('id', None) for _, demo in top_safe]
    unsafe_demo_ids = [demo.get('id', None) for _, demo in top_unsafe]
    knn_retrieval_time = time.time() - knn_start_time
    print(f"[KNN Retrieval] Time: {knn_retrieval_time:.4f}s")

    safe_examples = "\n\n".join([
        f"[SAFE CODE EXAMPLE {i+1}]:\n{demo.get('safe_code', '')}" for i, (_, demo) in enumerate(top_safe)
    ])
    unsafe_examples = "\n\n".join([
        f"[UNSAFE CODE EXAMPLE {i+1}]:\n{demo.get('unsafe_code', '')}" for i, (_, demo) in enumerate(top_unsafe)
    ])
    
    demo_prompt = f"""
    Here are reference code snippets:

    {safe_examples}

    {unsafe_examples}

    """

    code_example_prompt = demo_prompt + """
    Use these examples to guide your analysis of the following code.
    
    ⚠️ Important: Please give appropriate weight to the CODE EXAMPLES — both the SAFE CODE EXAMPLES and UNSAFE CODE EXAMPLES are verified and trustworthy. The target code may be structurally similar to these examples. You should carefully compare its structure and safeguards against both the SAFE and UNSAFE examples to make an accurate judgment. Do not assume the code is insecure simply because it uses certain methods; instead, analyze whether it follows similar protective patterns as in the SAFE examples, or exhibits dangerous behaviors as in the UNSAFE ones.
    """

    print("code_example_prompt", code_example_prompt)
    
    constitution_summarizer_prompt = """
    You are a helpful assistant that summarizes code patterns.
    Based on the reference code I provided (They are verified and trustworthy), summarize what kinds of patterns are safe and what kinds of patterns are unsafe into constitutions.
    I will further use the constitution you generate to guide vulnerability detection.

    Only output the constitutions in the following format, Example Output:
    Constitution: 
    Safe Constitutions:
    1.Safe Constitution1
    2.Safe Constitution2
    ...
    Unsafe Constitutions:
    1.Unsafe Constitution1
    2.Unsafe Constitution2
    ...

    """

    # Record constitution model call time
    constitution_start_time = time.time()
    global constitution_summarizer
    if constitution_summarizer is None:
        # Fallback to Client if constitution_summarizer is not initialized
        constitution_summarizer = Client(constitution_summarizer_prompt)
    else:
        # Use the pre-initialized constitution_summarizer with the prompt
        # Note: Some clients may need the prompt in generate() call, not in __init__
        pass
    
    # Combine the prompt with demo_prompt for generation
    full_prompt = constitution_summarizer_prompt + "\n\n" + demo_prompt
    constitution_summarizer_output = constitution_summarizer.generate(full_prompt)
    constitution_end_time = time.time()
    constitution_time = constitution_end_time - constitution_start_time
    
    constitution_summarizer_result = "You can refer to the following constitutions to guide your analysis: " + "\n" + constitution_summarizer_output
    print("constitution_summarizer_result", constitution_summarizer_result)
    print(f"[Constitution Model] Time: {constitution_time:.4f}s")

    llm_analyzer = Client(goal_prompt + '\n' + focus_cwe_prompt + '\n' + step_by_step_prompt + '\n'+ context_prompt + '\n' +  output_prompt)
    initial_prompt = (
        "Here is the code you need to analyze:\n"
        f"{code}\n\n"
        "Please perform a deep vulnerability analysis."
    )

    llm_baseline_output = llm_analyzer.generate(initial_prompt)
    llm_with_code_example_output = llm_analyzer.generate(initial_prompt + code_example_prompt)
    return llm_baseline_output, llm_with_code_example_output, constitution_summarizer_result, safe_demo_ids, unsafe_demo_ids, constitution_time, knn_retrieval_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Testing Agent")
    parser.add_argument("--knowledge_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--knowledge_same_cwe", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution_model", type=str, default="gpt4o", help="Model to use for constitution generation (gpt4o, gpt5, qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, llama, together_llama, qwen, claude)")
    parser.add_argument("--dynamic_analyzer_model", type=str, default=None, help="Model for dynamic testing analyzer. Default: anthropic/claude-sonnet-4-6 via OpenRouter. Set to match --model for fair model dependency experiments.")
    parser.add_argument("--dynamic_analyzer_local", action="store_true", default=False, help="Use local MetaLlamaClient for dynamic testing analyzer.")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index (inclusive) for data slicing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive) for data slicing")
    args = parser.parse_args()

    # Dynamically select the Client class (for the test model)
    Client = build_client(args.model)

    # Dynamically select the Constitution Client class
    constitution_sys_prompt = "You are a helpful assistant that summarizes code patterns."
    constitution_summarizer = build_client(args.constitution_model)(constitution_sys_prompt)
    
    print(f"[INFO] Using constitution model: {args.constitution_model}")

    with open(args.test_file, "r") as f:
        test_data = json.load(f)

    # Slice data if start_idx/end_idx specified (for parallel sharding)
    if args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx or 0
        e = args.end_idx or len(test_data)
        test_data = test_data[s:e]
        print(f"[INFO] Sliced test data to indices [{s}:{e}], {len(test_data)} items")

    with open(args.knowledge_file, "r") as f:
        knowledge_data = json.load(f)

    # Pre-initialize local dynamic analyzer to avoid reloading model every item

    for item in test_data:
        # Record total processing time for this test case
        case_start_time = time.time()
        
        item_id = item.get("id", "")
        code = item.get("llm_code", "")
        code_embedding = item.get("llm_code_embedding", [])
        cwe = item.get("cwe", "")
        code_type = item.get("code_example_code_type", "")

        if id_already_in_json(args.result_file, item_id):
            print(f"Skipping already processed ID: {item_id}")
            continue

        all_cwes = set()
        if cwe:
            all_cwes.add(cwe)
        related_cwes = list(all_cwes)
        focus_cwe_prompt = extract_focus_cwe_prompt(related_cwes, fixed_cwe_set_prompt)
        print(f"[Focus CWE Prompt for ID {item_id}]\n{focus_cwe_prompt}\n")
        
        llm_baseline_output, llm_with_code_example_output, constitution_summarizer_result, safe_demo_ids, unsafe_demo_ids, constitution_time, knn_retrieval_time = knn_testing(test_data,knowledge_data, code, code_embedding, cwe, focus_cwe_prompt,args.topk,args.knowledge_same_cwe)

        item["constitution_summarizer_result"] = constitution_summarizer_result
        item["safe_demo_ids"] = safe_demo_ids
        item["unsafe_demo_ids"] = unsafe_demo_ids
        
        result = dynamic_testing_with_constitution_analysis(code, code_type, constitution_summarizer_result)
        item["dynamic_testing_with_constitution_trajectory"] = trajectory[:]

        item["llm_baseline_output"] = llm_baseline_output
        llm_baseline_correctness="no vulnerabilities found" not in item['llm_baseline_output'].lower() if item.get("target", 1) == 1 else "no vulnerabilities found" in item['llm_baseline_output'].lower()
        item["llm_baseline_correctness"] = llm_baseline_correctness

        item["llm_with_code_example_output"] = llm_with_code_example_output
        llm_with_code_example_correctness="no vulnerabilities found" not in item['llm_with_code_example_output'].lower() if item.get("target", 1) == 1 else "no vulnerabilities found" in item['llm_with_code_example_output'].lower()
        item["llm_with_code_example_correctness"] = llm_with_code_example_correctness

        item["llm_with_constitution_output"] = result["llm_with_constitution_output"]
        llm_with_constitution_correctness="no vulnerabilities found" not in item['llm_with_constitution_output'].lower() if item.get("target", 1) == 1 else "no vulnerabilities found" in item['llm_with_constitution_output'].lower()
        item["llm_with_constitution_correctness"] = llm_with_constitution_correctness
        
        dynamic_testing_response = result["dynamic_testing_response"] # use for dynamic_without_constitution
        dynamic_testing_results_time_cost = result.get("dynamic_testing_results_time_cost", 0.0)
        # Get the result first; default to (None, None) if it is None
        dt_code, dt_exec = result.get("dynamic_testing_results") or (None, None)
        item["dynamic_testing_code"] = dt_code
        item["dynamic_testing_execution_results"] = dt_exec
        item["dynamic_constitution_final_judgment"] = result["dynamic_constitution_final_judgment"]

        final_judgment_lower = result['dynamic_constitution_final_judgment'].lower()        
        item["dynamic_constitution_final_judgment_correctness"] = evaluate_static_dynamic_final_judgment(final_judgment_lower, item)

        dynamic_without_constitution_result = dynamic_testing_without_constitution_analysis(
            code,
            code_type,
            llm_baseline_output,
            dynamic_testing_response,
            result.get("dynamic_testing_results"),
            dynamic_testing_results_time_cost,
        )
        
        item["dynamic_without_constitution_final_judgment"] = dynamic_without_constitution_result["dynamic_without_constitution_final_judgment"]
        dynamic_without_constitution_final_judgment_lower = dynamic_without_constitution_result['dynamic_without_constitution_final_judgment'].lower()        
        item["dynamic_without_constitution_final_judgment_correctness"] = evaluate_static_dynamic_final_judgment(dynamic_without_constitution_final_judgment_lower, item)
        item["dynamic_testing_without_constitution_trajectory"] = trajectory[:]

        # Record total processing time
        case_end_time = time.time()
        total_time = case_end_time - case_start_time
        
        # Add timing information
        item["knn_retrieval_time"] = round(knn_retrieval_time, 4)
        item["constitution_model_time"] = round(constitution_time, 4)
        item["total_processing_time"] = round(total_time, 4)
        
        item.pop("llm_code_embedding")
        item.pop("safe_embedding")
        item.pop("unsafe_embedding")
        record_to_json(
            filename=args.result_file,
            **item
        )
        
        print(f"[ID {item_id}] Timing - KNN: {knn_retrieval_time:.4f}s, Constitution: {constitution_time:.4f}s, Total: {total_time:.4f}s")
