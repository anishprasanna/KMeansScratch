import java.util.Scanner;
public class OrderedIntegers {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);

        int a;
        int b;
        int c;

        System.out.println("Enter three integers: ");
        a = scnr.nextInt();
        b = scnr.nextInt();
        c = scnr.nextInt();

        if (a <= b && b <= c){
            System.out.println("The integers in decreasing order are: " + c + " " + b + " " +a);
        }
        else if (a <= c && c <= b) {
            System.out.println("The integers in decreasing order are: " +b + " " + c + " " + a);
        }
        else if (b <= a && a <= c) {
            System.out.println("The integers in decreasing order are: " +c + " " + a + " " + b);
        }
        else if (b <= c && c <= a) {
            System.out.println("The integers in decreasing order are: " +a + " " + c + " " + b);
        }
        else if (c <= a && a <= b) {
            System.out.println("The integers in decreasing order are: " +b + " " + a + " " + c);
        }
        else if (c <= b && b <=a) {
            System.out.println("The integers in decreasing order are: " +a + " " + b + " " + c);
        }





    }
}
