import java.util.Scanner;


public class CountInputLength
{
    public static void main (String [] args)
    {
        //creates variables
        Scanner scnr = new Scanner(System.in);
        String userString;
        int numChar = 0;
        System.out.println("Enter a line of text: ");
        userString = scnr.nextLine();

        // gets length of string to count characters
        numChar = userString.length();
        //Runs for loop to loop through string

        for( int i = 0; i < userString.length(); i++)
        {

           // Checks to See if the character is a space or a period or a comma
            if( userString.charAt(i) == 44 || userString.charAt(i) == 32 || userString.charAt(i) == 46)
                //subtracts the amount of characters that are spaces, periods, or commas
                numChar = numChar - 1;


        }
        System.out.println("The text length without spaces, periods, or commas is: " + numChar);



    }
}
