import java.util.Scanner;
public class CountInputLength {
    public static void main(String [] arg)
    {
        Scanner scnr = new Scanner(System.in);              //Set up for user to input

        String userText;                         //Variable for user's input
        int numLetters;                         //Variable for number of letter
        int i;                                 //Variable i for the for loop

        System.out.println("Count Input Length =============");         //Header
        System.out.println("");                                         //Empty line

        System.out.print("Enter a line of text: ");                    //Statement outputted for user to answer
        userText = scnr.nextLine();                                   //To allow user's input
        numLetters = 0;                                              //Assigning numLetters up to zero

        for (i = 0; i < userText.length(); ++i) {                  //A for loop to compute the length of user's input without special characters
            if (Character.isLetter(userText.charAt(i))) {
                numLetters += 1;
            }
        }
            System.out.println("The text length without spaces, periods, or commas is: " + numLetters);     //Output statement and number of letters to screen

    }
}
