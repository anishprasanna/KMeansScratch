import java.util.Scanner;
public class CountInputLength {
    public static void main(String [] args )
    {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Count Input Length =============");
        System.out.println("");

        //Define variables
        String userString;
        int userStringLength;
        int i;
        int newStringLength = 0;

        // Ask for user input
        System.out.print("Enter a line of text: ");
        userString = scnr.nextLine();
        userStringLength = userString.length();

        //Take out spaces
        for( i = 0; i < userStringLength; i++) {
            if ( (userString.charAt(i) != ' ' ) && ( userString.charAt(i) != '.' ) && ( userString.charAt(i) != ',' ))  {
                newStringLength += 1;
            }
        }

        //Print new string length
        System.out.println("The text length without spaces, periods, or commas is: " + newStringLength);
    }
}
