import java.util.Scanner;

public class CountInputLength {
    public static void main (String [] args) {
        int userStringLength;
        int originalUserStringLength = 0;
        String userInputString;
        int programCounter = 0;
        boolean charCheck = true;
        //These are the variables I used. It includes the string length, original string length, input string, a counter and a boolean telling the program when to stop checking the string

        Scanner scnr = new Scanner(System.in);
        System.out.println("Count Input Length =============");
        System.out.println("");
        System.out.print("Enter a line of text: ");
        userInputString = scnr.nextLine();
        originalUserStringLength = userInputString.length();
        userStringLength = originalUserStringLength;
        //This code block prints out the prompt to the user and then records their input and the length of said input

        while (charCheck) {
            if (userInputString.charAt(programCounter) == ' '){
                userStringLength --;
                programCounter ++;
            }
            //This code detects if there is a space and then subtracts 1 from the length and adds to the counter.
            else if (userInputString.charAt(programCounter) == ','){
                userStringLength --;
                programCounter ++;
            }
            //This code detects if there is a comma and then subtracts 1 from the length and adds to the counter.
            else if (userInputString.charAt(programCounter) == '.'){
                userStringLength --;
                programCounter ++;
            }
            //This code detects if there is a period and then subtracts 1 from the length and adds to the counter.
            else {
                programCounter ++;
            }
            //THis just goes when the program finds no commas, spaces, or periods.
            if (programCounter < originalUserStringLength){
                charCheck = true;
            }
            else{
                charCheck = false;
            }
            //This if statement tells the program to stop when need be
        }
        System.out.println("The text length without spaces, periods, or commas is: " + userStringLength);
        //This prints the final product.
    }
}
