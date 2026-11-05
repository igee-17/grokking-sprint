"""Hidden operations. Never ship this file to an agent workspace.

Calibration notes:
  * quad (a^2 + ab + b^2) was dropped: no tried config generalized in budget,
    so it would measure luck rather than method.
"""
HIDDEN = {
    "mul": lambda a, b, p: (a * b) % p,
    "aff": lambda a, b, p: (a + 5 * b) % p,
    "div": lambda a, b, p: (a * pow(int(b), p - 2, p)) % p if b != 0 else None,
}
