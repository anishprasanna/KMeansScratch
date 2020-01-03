import java.util.Scanner;

public class DecimalToBinary {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);
        int userInput;
        int remainder = 0;
        int lastIndex;
        int revLength;
        int i = 0;
        String binary = "";
        String notReversed = "";

        System.out.println("Decimal 2 Binary =====================");
        System.out.print("\nEnter an integer: ");
        userInput = scnr.nextInt();

        /*
        This loop takes the remainder of the userInput and divides userInput by 2.
        The remainder is added to an empty string every time.
        The userInput will be divided by 2 until it hits 0. The remainder of each quotient
        will be added to the empty string as well.
         */

        while(userInput > 0)
        {
            remainder = userInput % 2;
            userInput = userInput / 2;
            notReversed += remainder;
        }

        revLength = notReversed.length();

        /*
        This loop iterates through the string and returns each character in reverse order
        until it has iterated through the entire string.
         */

        for (i = 1; i <= revLength; ++i) {

            System.out.print(notReversed.charAt(notReversed.length() - i));
        }

    }
}
