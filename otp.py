import base64

class OneTimePad:
    def __init__(self, key):
        self.key = key

    def encrypt(self, message):
        """Encrypts the message using a One-Time Pad."""
        message_bytes = message.encode()
        cipher_bytes = bytes([m ^ k for m, k in zip(message_bytes, self.key)])
        return base64.b64encode(cipher_bytes).decode(), base64.b64encode(self.key).decode()

    def decrypt(self, cipher_text):
        """Decrypts a One-Time Pad encrypted message."""
        cipher_bytes = base64.b64decode(cipher_text)
        key_bytes = base64.b64decode(self.key)
        message_bytes = bytes([c ^ k for c, k in zip(cipher_bytes, key_bytes)])
        return message_bytes.decode()

# Example Usage
otp = OneTimePad(use_qrng=True)  # Use Quantum RNG for the key
message = "Hello, Quantum Security!"

# Encrypt
cipher_text, key = otp.encrypt(message)
print(f"Cipher Text: {cipher_text}")
print(f"Key: {key}")

# Decrypt
decrypted_message = otp.decrypt(cipher_text, key)
print(f"Decrypted Message: {decrypted_message}")
