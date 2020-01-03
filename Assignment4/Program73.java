import java.util.Scanner;

public class CountInputLength {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String textInput;
        int textLength;

//Count Input Length =============
        System.out.println("Count Input Length =============");

//Ask user to enter a line of text
        System.out.println("Enter a line of text: ");
        textInput = sc.nextLine();

//Remove spaces, periods, commas by the replace method
        textInput = textInput.replace(" ", "");
        textInput = textInput.replace(",", "");
        textInput = textInput.replace(".", "");

// Use the for-loop to count the text length.
        textLength = textInput.length();
        for (int i = 0; i > -2; --i) {
        }

//Output the text length without spaces, periods, or commas
        System.out.print("The text length without spaces, periods, or commas is: " + textLength);

    }
}
