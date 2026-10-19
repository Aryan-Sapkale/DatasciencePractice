import math

def show_menu():
    print("\n" + "=" * 35)
    print("       SMART PYTHON CALCULATOR")
    print("=" * 35)
    print("1. Addition (+)")
    print("2. Subtraction (-)")
    print("3. Multiplication (*)")
    print("4. Division (/)")
    print("5. Power (**)")
    print("6. Square Root")
    print("7. Exit")
    print("=" * 35)


while True:
    show_menu()
    choice = input("Choose an option (1-7): ")

    if choice == "7":
        print("Thank you for using Smart Calculator!")
        break

    if choice not in ("1", "2", "3", "4", "5", "6"):
        print("Invalid choice! Please select 1-7.")
        continue

    try:
        if choice == "6":
            number = float(input("Enter a number: "))

            if number < 0:
                print("Error: Negative numbers have no real square root.")
            else:
                print("Square root =", math.sqrt(number))

        else:
            a = float(input("Enter first number: "))
            b = float(input("Enter second number: "))

            if choice == "1":
                result = a + b
            elif choice == "2":
                result = a - b
            elif choice == "3":
                result = a * b
            elif choice == "4":
                if b == 0:
                    print("Error: Cannot divide by zero.")
                    continue
                result = a / b
            elif choice == "5":
                result = a ** b

            print("Result =", result)

    except ValueError:
        print("Invalid input! Please enter numeric values.")
