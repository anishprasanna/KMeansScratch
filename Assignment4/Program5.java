import java.util.Scanner;
public class CountInputLength {
    public static void main(String [] args)
    {
        Scanner scnr = new Scanner (System.in);

        //Declare the variables
        String a;
        int charPosition;
        int charCount = 0;

        //Prompt the user to enter a string and read the string
        System.out.println("Count Input Length =============");
        System.out.println();
        System.out.println("Enter a line of text: ");
        a = scnr.nextLine();

        //Use for loop to count the text length including only letters and digits (no punctuation or spaces)
        for (charPosition = 0; charPosition < a.length(); charPosition++){
            if ( (a.substring(charPosition, charPosition + 1).equals(" ") || a.substring(charPosition, charPosition + 1).equals(".") || a.substring(charPosition, charPosition + 1).equals(",")) ) {
                charCount = charCount + 0;
            }
            else {
                charCount = charCount + 1;
            }
        }

        //Print text length
        System.out.println("The text length without spaces, periods, or commas is: " + charCount);

    }
}
