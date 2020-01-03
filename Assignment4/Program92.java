import java.util.Scanner;

public class ReplaceTester {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);
        String userInput;
        String character1;
        String character2;
        String newString;

        System.out.print("Enter a string that you wish to modify: ");
        userInput = scnr.nextLine();
        System.out.print("Enter a character that you want to replace: ");
        character1 = scnr.nextLine();
        System.out.print("Enter a replacement character: ");
        character2 = scnr.nextLine();

        //changing string
        if (userInput.indexOf(character1) == -1)
        {
            System.out.print("The character you want to replace does not exist the string you gave. Please choose a character in the string.");
        }
        else
        {
            newString = userInput.replace(character1, character2);

            System.out.print("The modified string is: " + newString);
        }




    }
}
