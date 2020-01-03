import java.util.Scanner;

public class CountInputLength {

    public static void main(String [] args){

        Scanner scnr = new Scanner(System.in);
        String userText;
        int numCharacters; // Defines the number of characters entered
        int i; //Loop variable

        System.out.print("Enter a line of text: "); // Prompts user to enter a line of text
        userText = scnr.nextLine();


        numCharacters = 0;
        for (i = 0; i < userText.length(); ++i) { // Creates a loop that iterates a number of times and counts the characters
            if (Character.isLetter(userText.charAt( i ))) {
             numCharacters += 1;

            }

        }

        System.out.print("The text length without spaces, periods, or commas: " + i  ); // Outputs number of characters









    }
}
