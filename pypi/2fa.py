import pyotp

key = 'WPQOQV4KBAJEYMUO3G7VAYX2CENVSZBI'
totp = pyotp.TOTP(key)
print(totp.now())