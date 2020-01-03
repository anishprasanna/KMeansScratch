import java.util.Scanner;

public class CountInputLength {
    public static void main (String [] args) {
        Scanner scnr = new Scanner(System.in);
        //set up scanner

        String userInput;
        int textLength;
        int i;
        //define my variables

        System.out.println("Count Input Length =============");
        System.out.println(" ");
        System.out.println("Enter a line of text: ");

        userInput = scnr.nextLine();

        textLength = 0;
        for( i = 0 ; i < userInput.length() ; ++ i )   {
            if (Character.isLetter( userInput.charAt(i) ) || Character.isDigit( userInput.charAt(i)))   {
             textLength += 1;
            }
        }
        //counts only characters that are a letter or number

        System.out.println("The text length without spaces, periods, or commas is: " + textLength);

    }
}
