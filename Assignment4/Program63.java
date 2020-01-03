import java.util.Scanner;
public class DecimalToBinary {
    public static void main(String[] args) {
        System.out.println("Decimal 2 Binary =====================");
        Scanner scnr = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        //set variables to record the integer the user types in, variable for loop, and prepare string to record the output
        int number1 = scnr.nextInt();
        int i;
        String string1 = "";
        // loop to get the modular number to add to the string
        while (number1 > 0) {
            int number2 = number1 % 2;
            //I add each new number before the string
            string1 = number2 + string1;
            number1 = number1 / 2;
        }
        //type out the outputs
        System.out.println("The binary equivalent is: " + string1);

    }
}
