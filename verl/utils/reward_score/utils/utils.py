"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import os
import os
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Optional


def _strip_tex_wrappers(text: str) -> str:
    """Remove common LaTeX wrappers (e.g. \text{}) so \boxed{\text{A}} is recognized as A."""
    if not text:
        return text
    patterns = [
        r"\\text\s*\{([^}]*)\}",
        r"\\mathrm\s*\{([^}]*)\}",
        r"\\operatorname\s*\{([^}]*)\}",
        r"\\mathbf\s*\{([^}]*)\}",
        r"\\mathsf\s*\{([^}]*)\}",
        r"\\mathit\s*\{([^}]*)\}",
        r"\\mathbb\s*\{([^}]*)\}",
        r"\\mathtt\s*\{([^}]*)\}",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, r"\1", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\s*", " ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ").replace("_", " ")
    return " ".join(cleaned.split())


def _strip_tex_wrappers(text: str) -> str:
    """Remove common LaTeX wrappers such as \\text{} or \\mathrm{} from boxed content."""
    if not text:
        return text
    patterns = [
        r"\\text\s*\{([^}]*)\}",
        r"\\mathrm\s*\{([^}]*)\}",
        r"\\operatorname\s*\{([^}]*)\}",
        r"\\mathbf\s*\{([^}]*)\}",
        r"\\mathsf\s*\{([^}]*)\}",
        r"\\mathit\s*\{([^}]*)\}",
        r"\\mathbb\s*\{([^}]*)\}",
        r"\\mathtt\s*\{([^}]*)\}",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, r"\1", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\s*", " ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ").replace("_", " ")
    return " ".join(cleaned.split())


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def extract_answer(raw_output: str, is_mcq: bool = True, debug: bool = False) -> Optional[str]:
    """
    Extracts the final answer from the model's raw output using a prioritized regex strategy.
    It prioritizes finding the *last* occurrence of a pattern, assuming it's the model's final conclusion.
    This function matches the logic from evaluate_models.py exactly.
    """
    # Temp debug: capture all "Therefore, the correct answer is X" cases
    temp_debug_patterns = {
        'A': "Therefore, the correct answer is A" in raw_output,
        'B': "Therefore, the correct answer is B" in raw_output,
        'C': "Therefore, the correct answer is C" in raw_output,
        'D': "Therefore, the correct answer is D" in raw_output,
        'E': "Therefore, the correct answer is E" in raw_output,
    }
    temp_expected_answer = None
    for answer, found in temp_debug_patterns.items():
        if found:
            temp_expected_answer = answer
            break
    # Priority 1: Find all occurrences of \boxed{...} and use the last one.
    # Try double backslash first (most common in actual model outputs), then single
    box_matches = re.findall(r"\\\\boxed\{(.+?)\}", raw_output, re.DOTALL)
    if not box_matches:
        box_matches = re.findall(r"\\boxed\{(.+?)\}", raw_output, re.DOTALL)
    if debug:
        print(f"[DEBUG] Box matches found: {box_matches}")
    if box_matches:
        # Get the content of the last found box
        last_box_content = box_matches[-1].strip()
        use_deepmath_fix = os.environ.get("VERL_DEEPMATH_FORMAT_FIX", "0") == "1"
        simplified_content = _strip_tex_wrappers(last_box_content) if use_deepmath_fix else last_box_content
        if debug:
            print(f"[DEBUG] Found boxed content: {last_box_content}")
            if use_deepmath_fix:
                print(f"[DEBUG] Simplified boxed content: {simplified_content}")
        if is_mcq:
            # For MCQ, find the single letter answer within the box content.
            # Check for a letter directly in the box, e.g., \boxed{A}
            letter_match = re.search(r"^\s*([A-E])\s*$", simplified_content, re.IGNORECASE)
            if letter_match:
                if debug:
                    print(f"[DEBUG] Extracted from box (direct): {letter_match.group(1)}")
                return letter_match.group(1).upper()
            # If the box contains more text, find the letter within it.
            letter_match_inside = re.search(r"\b([A-E])\b", simplified_content, re.IGNORECASE)
            if letter_match_inside:
                if debug:
                    print(f"[DEBUG] Extracted from box (inside): {letter_match_inside.group(1)}")
                return letter_match_inside.group(1).upper()
        else:
            # For fill-in-the-blank, the content of the box is the answer.
            return last_box_content

    # Priority 2 (Fallback for MCQs): Look for the last explicit answer statement.
    if is_mcq:
        use_deepmath_fix = os.environ.get("VERL_DEEPMATH_FORMAT_FIX", "0") == "1"
        if use_deepmath_fix:
            answer_patterns = [
                r"(?:the|my)\s+(?:\w+\s+)?answer\s+is\s*:?\s*(?:\\boxed\{(?:\\text\{)?\s*([A-E])\s*\}*\}|([A-E]))",
                r"(?:the|my)\s+(?:\w+\s+)?final\s+answer\s+is\s*:?\s*(?:\\boxed\{(?:\\text\{)?\s*([A-E])\s*\}*\}|([A-E]))",
                r"is\s*:\s*(?:\\boxed\{(?:\\text\{)?\s*([A-E])\s*\}*\}|([A-E]))",
                r"option\s+([A-E])\b"
            ]
        else:
            answer_patterns = [
                r"(?:the|my)\s+(?:\w+\s+)?answer\s+is\s*:?\s*\b([A-E])\b",
                r"(?:the|my)\s+(?:final\s+)?answer\s+is\s*:?\s*\b([A-E])\b",
                r"is\s*:\s*\b([A-E])\b"
            ]
        
        last_match_pos = -1
        found_answer = None
        all_matches = []
        
        for pattern in answer_patterns:
            for match in re.finditer(pattern, raw_output, re.IGNORECASE):
                if use_deepmath_fix:
                    groups = match.groups()
                    candidate = None
                    for grp in groups:
                        if grp and grp.strip():
                            candidate = grp.strip().upper()
                            break
                else:
                    candidate = match.group(1).upper() if match.group(1) else None
                all_matches.append((match.start(), candidate, pattern, match.group(0)))
                if candidate and match.start() > last_match_pos:
                    last_match_pos = match.start()
                    found_answer = candidate

        if debug and all_matches:
            print(f"[DEBUG] All answer pattern matches: {all_matches}")
            print(f"[DEBUG] Final extracted answer: {found_answer}")

        if found_answer:
            # Temp debug: if this is a problematic case and the wrong answer was returned
            if temp_expected_answer and found_answer != temp_expected_answer:
                print(f"[TEMP DEBUG] Found 'Therefore, the correct answer is {temp_expected_answer}' but returning '{found_answer}'")
                print(f"[TEMP DEBUG] All matches: {all_matches}")
                print(f"[TEMP DEBUG] Full text: {repr(raw_output[:500])}...")  # only show first 500 chars
            return found_answer

    # If no reliable pattern is matched, return None. We avoid broad fallbacks
    # like "find the last capital letter" to prevent coincidental extractions.
    if debug:
        print(f"[DEBUG] No answer pattern matched")

    # Temp debug: if this is a problematic case, print detailed info
    if temp_expected_answer:
        print(f"[TEMP DEBUG] Found 'Therefore, the correct answer is {temp_expected_answer}' but returning None")
        print(f"[TEMP DEBUG] Full text: {repr(raw_output[:500])}...")  # only show first 500 chars
    
    return None

# Keep old function for backward compatibility but redirect to new one
def extract_answer_old(passage: str) -> str:
    """Legacy function for backward compatibility"""
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth, is_mcq=True)
    given_answer = extract_answer(solution_str, is_mcq=True)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) \
        or grade_answer_sympy(given_answer, ground_truth)
