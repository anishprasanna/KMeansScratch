import java.util.Scanner;
public class CountInputLength {
    public static void main(String[] args) {
        String userInput;
        int total = 0;
        //I initialized the total at 0.
        Scanner scnr = new java.util.Scanner(System.in);
        //I will now ask the user for their input.
        System.out.println("Enter a line of text: ");
        userInput = scnr.nextLine();

        // The for loop will run through the string (inputted by the user) character by character.
        for (int i = 0; i < userInput.length(); i++) {
            char c = userInput.charAt(i);
            //If the character is a space, period, or comma, a value will not be added to the total.
            if (c == ' ') {
            } else if (c == ',') {
            } else if (c == '.') {
            } else {
                //If the character is NOT a space, a value will be added to the total.
                total = total + 1;
            }

        }
        System.out.println("The total length without spaces, periods, or commas is: " + total);

        }
    }
