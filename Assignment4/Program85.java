import java.util.Scanner;
public class ReplaceTester {
    public static void main (String [] args)
    {
        Scanner scnr = new Scanner (System.in);

        //Declare variables
        String stringToEdit;
        String editedString;
        char character1;
        char character2;

        // Print the purpose fo the program then request for a string to be entered
        System.out.println("Replace Text=====================");
        System.out.println();
        System.out.println("Enter a string that you want to modify: ");

        //Read the string
        stringToEdit = scnr.nextLine();

        //Ask for the character to replace and the character to substitute
        System.out.println("Enter a character that you want to replace: ");
        character1 = scnr.next().charAt(0);
        System.out.println("Enter a replacement character: ");
        character2 = scnr.next().charAt(0);

        //Edit the string and print the modified string
        editedString = stringToEdit.replace(character1, character2);
        System.out.println(editedString);





    }
}
