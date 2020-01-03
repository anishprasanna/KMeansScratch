import java.util.Scanner;

public class CountInputLength {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);
        String userInput;
        System.out.println("Count Input Length ============= ");
        System.out.print("\nEnter a line of text: ");
        userInput = scnr.nextLine();
        userInput = userInput.replace(" ", "");
        userInput = userInput.replace(",", "");
        userInput = userInput.replace(".", "");
        System.out.print("The text length without spaces, periods, or commas is: " + userInput.length());

    }
}
