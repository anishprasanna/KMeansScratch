import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Input three integers: ");
        int a = sc.nextInt();
        int b = sc.nextInt();
        int n = sc.nextInt();

        for (int i = 1; i <= n; i++) {
            if (i % a == 0)
                System.out.print("Fizz");

            if (i % b == 0)
                System.out.print("Buzz");

            if ((i % a != 0) && (i % b != 0))
                System.out.print(i);

            System.out.println();

        }
    }
}
