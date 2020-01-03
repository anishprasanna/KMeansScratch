import java.util.Scanner;

public class DecimalToBinary {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int input;

        //Decimal 2 Binary =====================
        System.out.println("Decimal 2 Binary =====================\n");

        //Ask user for the integer
        System.out.print("Enter an integer: ");
        input = sc.nextInt();

        int n = input; //43
        int remainder = 0;
        int binary = 0;
        int i = 0;

        while (n != 0){
            remainder = input % 2;
            n = input / 2;
            input = n;

            binary = binary + remainder * (int)Math.pow(10,i);
            i = i + 1;
        }

        System.out.println("The binary equivalent is: " + binary);


        //
    }
}
