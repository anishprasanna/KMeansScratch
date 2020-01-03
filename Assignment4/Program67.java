import java.util.Scanner;
public class DecimalToBinary {
    public static void main(String[] args) {
        // Output program title
        System.out.println("Decimal 2 Binary =====================\n");

        // Read user input
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int userInteger = sc.nextInt();

        // Only accept positive integers as a valid input
        while (userInteger < 0) {
            System.out.println("Invalid input. Please enter a positive integer.");
            System.out.print("Enter an integer: ");
            userInteger = sc.nextInt();
        }

        // Convert integer from decimal to binary (for all digits except for the last binary digit)
        int binaryInput;
        String reversedBinary = "";
        String correctBinary = "";

        while ( !((userInteger / 2) == 0 )) {
            binaryInput = userInteger % 2;
            userInteger = userInteger / 2;
            reversedBinary += binaryInput;
        }
        // Convert from decimal to binary for the last binary digit
        binaryInput = userInteger % 2;
        userInteger = userInteger / 2;
        reversedBinary += binaryInput;

        // Reverse order of the string
        for (int i = reversedBinary.length() -1; i >= 0; i--) {
            correctBinary += reversedBinary.charAt(i);
        }
        System.out.println("The binary equivalent is: " + correctBinary);
    }
}
