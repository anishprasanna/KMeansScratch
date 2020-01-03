import java.util.Scanner;
public class CountInputLength {
    public static void main(String[] args) {

        // Output program title
        System.out.println("Count Input Length =============\n");

        // Read a line of user input
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a line of text: ");
        String userText = sc.nextLine();

        // Count number of characters excluding spaces, periods, or commas
        int numCharacters = 0;
        for (int i = 0; i < userText.length(); i++) {
            if ( (!Character.isWhitespace(userText.charAt(i))) && (userText.charAt(i) != '.') && (userText.charAt(i) != ',')) {
                numCharacters += 1;
            }
        }
        // Output number of characters excluding spaces, periods, or commas
        System.out.println("The text length without spaces, periods, or commas is: " + numCharacters);
    }
}
