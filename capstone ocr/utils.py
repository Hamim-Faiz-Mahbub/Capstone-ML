import string

def create_custom_alphabet(labels, default_alphabet=""):
    """
    Membuat alphabet kustom berdasarkan label yang diberikan dan alphabet default.
    """
    if not labels and not default_alphabet:
        
        return string.digits + string.ascii_lowercase + string.ascii_uppercase + " .,:()/-%&!#"

    all_chars_from_labels = "".join(str(label) for label in labels)
    unique_chars = set(all_chars_from_labels)

    if default_alphabet:
        for char_default in default_alphabet:
            unique_chars.add(char_default)

    custom_alphabet = "".join(sorted(list(unique_chars)))
    return custom_alphabet