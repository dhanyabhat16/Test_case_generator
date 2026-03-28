"""
sample_code.py
--------------
Sample Python file used to test the code parser and embedder.
Contains a mix of simple and complex functions across a mini banking module.
"""


# ─── Simple math utilities ───────────────────────────────────────────────────

def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """
    Divide a by b and return the result.

    Args:
        a: Numerator.
        b: Denominator. Must not be zero.

    Returns:
        Result of a / b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


# ─── String utilities ─────────────────────────────────────────────────────────

def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (case-insensitive).

    Args:
        s: Input string.

    Returns:
        True if s is a palindrome, False otherwise.
    """
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    """Return the number of words in a string."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to max_length characters, appending suffix if truncated.

    Args:
        text: The input string.
        max_length: Maximum allowed length.
        suffix: String to append when truncated. Defaults to '...'.

    Returns:
        Truncated string with suffix, or original if within limit.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ─── Banking domain ──────────────────────────────────────────────────────────

class BankAccount:
    """Simple bank account with deposit, withdraw, and transfer operations."""

    def __init__(self, account_id: str, owner: str, initial_balance: float = 0.0):
        self.account_id = account_id
        self.owner = owner
        self.balance = initial_balance
        self.transactions = []

    def deposit(self, amount: float) -> float:
        """
        Deposit funds into the account.

        Args:
            amount: Amount to deposit. Must be positive.

        Returns:
            New balance after deposit.

        Raises:
            ValueError: If amount is not positive.
        """
        if amount <= 0:
            raise ValueError(f"Deposit amount must be positive. Got: {amount}")
        self.balance += amount
        self.transactions.append({"type": "deposit", "amount": amount})
        return self.balance

    def withdraw(self, amount: float) -> float:
        """
        Withdraw funds from the account.

        Args:
            amount: Amount to withdraw. Must be positive and <= balance.

        Returns:
            New balance after withdrawal.

        Raises:
            ValueError: If amount is invalid or exceeds balance.
        """
        if amount <= 0:
            raise ValueError(f"Withdrawal amount must be positive. Got: {amount}")
        if amount > self.balance:
            raise ValueError(
                f"Insufficient funds. Balance: {self.balance}, Requested: {amount}"
            )
        self.balance -= amount
        self.transactions.append({"type": "withdrawal", "amount": amount})
        return self.balance

    def get_balance(self) -> float:
        """Return the current account balance."""
        return self.balance

    def get_transaction_count(self) -> int:
        """Return the number of transactions made on this account."""
        return len(self.transactions)


def calculate_interest(principal: float, rate: float, years: int) -> float:
    """
    Calculate simple interest.

    Args:
        principal: Initial amount.
        rate: Annual interest rate as a decimal (e.g., 0.05 for 5%).
        years: Number of years.

    Returns:
        Total interest earned (not total amount).

    Raises:
        ValueError: If any argument is negative.
    """
    if principal < 0 or rate < 0 or years < 0:
        raise ValueError("Principal, rate, and years must all be non-negative.")
    return principal * rate * years


def validate_email(email: str) -> bool:
    """
    Basic email format validation.

    Args:
        email: Email address string to validate.

    Returns:
        True if the email looks valid, False otherwise.
    """
    import re
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def fibonacci(n: int) -> list:
    """
    Generate the first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci numbers to generate. Must be >= 0.

    Returns:
        List of the first n Fibonacci numbers.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq