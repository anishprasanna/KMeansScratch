import java.util.Scanner;

public class CountInputLength {
    public static void main (String[] args) {

        // Creates scanner for user input
        Scanner scnr = new Scanner(System.in);

        // Declare variables and initializes those that need to be
        String userString;
        int reducedLength = 0;

        // Write welcome message
        System.out.println("Count Input Length ===============");

        // Write prompt for user input
        System.out.print("Enter a line of text: ");

        // Read user's string
        userString = scnr.nextLine();

        // For loop checks each character to see if it should be included in the count. If it should, adds one to count
        for (int i = 0; i < userString.length(); i++) {
            if (userString.substring(i, i + 1).equals(" ") || userString.substring(i, i + 1).equals(".")
                    || userString.substring(i, i + 1).equals(",")) {
                reducedLength += 0;
            }

            else {
                reducedLength += 1;
            }
        }

        // Writes reduced count (without unwanted characters)
        System.out.print("The text length without spaces, periods, or commas is: " + reducedLength);
    }
}
