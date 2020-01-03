import java.util.Scanner;
public class DecimalToBinary {
    public static void main(String[] arg) {
        //create scanner and variables
        Scanner scnr = new Scanner(System.in);
        System.out.println("Enter an integer to convert to binary: ");
        int newInteger = scnr.nextInt();
        String output = new String("");

        //calculate and print the binary code (code prints the correct binary, but is printed backwards)
        while (newInteger > 0) {
            if (newInteger % 2 == 0) {
                System.out.print(0);
            } else if (newInteger % 2 == 1) {
                System.out.print(1);
            }
            newInteger /= 2;
        }
    }
}
