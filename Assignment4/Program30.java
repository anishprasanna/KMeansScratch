import java.util.Scanner;
public class CountInputLength {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);

        // Define the variables to be used in this program
        String userText;
        int lengthText;
        int total=0;
        char c;

        // Ask the user to enter the data needed by the program
        System.out.println("Count Input Length ==============\n");
        System.out.print("Enter a line of text: ");
        userText = scnr.nextLine();

        // Do the needed calculations, use variables to store your results
        for(int i =0; i<userText.length(); i++){
            c = userText.charAt(i);
            if(c != 32 && c!= 44 && c!= 46){
                total += 1;
            }
        }

        // Print the results to the screen
        System.out.print("The text length without spaces, periods, or commas is: " + total);

    }
}
