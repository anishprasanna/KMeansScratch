import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String []arg){
        Scanner scnr = new Scanner(System.in);
        int i=0;
        int x;
        int y;
        int n;

        System.out.println("Input three integers: " );
        x = scnr.nextInt();
        y = scnr.nextInt();
        n = scnr.nextInt();
        for (i=1; i <= n; i++) {
            if (((i % x) == 0 && ((i % y) == 0))) {
                System.out.println("FizzBuzz");
            } else if ((i % x) == 0) {
                System.out.println("Fizz");
            } else if ((i % y) == 0) {
                System.out.println("Buzz");
            } else
                System.out.println(i);
        }



    }
}
