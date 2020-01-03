import java.util.Scanner;
public class DecimalToBinary {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Decimal 2 Binary ==============\n");
        System.out.print("Enter an integer: ");
        int num = scnr.nextInt();
        System.out.print("The binary equivalent is: ");

        String rem = "";
        while (num/2 != 0 ){

            if(num%2 == 0){
                rem = rem + 0;
            }
            if(num%2 == 1){
                rem = rem + 1;
            }
            num = num / 2;

        }
        rem = rem + 1;

        for(int i= rem.length(); i>0; i--) {
            System.out.print(rem.charAt(i-1));
        }
    }
}

