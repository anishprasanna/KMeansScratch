import java.util.Scanner;

public class CountInputLength {
    public static void main (String [] args)
    {
        // ask the user to enter a string
        System.out.print("Enter a line of text: ");
        Scanner sc = new Scanner(System.in);
        String userString = sc.nextLine();

        // calculate how long the string is without whitespace or punctuation using a for loop
            // the loop will count any character other than punctuation, so numbers and digits

        int numCharacters = 0;
        int i;                                                          // counter variable

        for (i = numCharacters; i < userString.length(); ++i) {         // iterates from index 0 to the end
            if (Character.isLetterOrDigit(userString.charAt(i))) {      // only adds to numCharacters is letter or digit
                numCharacters += 1;
            }
        }

        // output the length of text without spaces or punctuation
        System.out.print("The text length without spaces, periods, or commas is: " + numCharacters);
    }

}
