import java.util.Scanner;
public class Program2 {
    public static void main (String[] args)
    {
        //get three integers from user
        Scanner scnr = new Scanner(System.in);
        System.out.print("Input three integers: ");

        int num1 = scnr.nextInt();
        int num2 = scnr.nextInt();
        int num3 = scnr.nextInt();
        String s = "";

        //for loop to find if each number is divisible by first and second
        for (int i = 1; i <= num3; i++)
        {
            s = "";
            if (i % num1 == 0)
            {
                s += "Fizz";
                if (i % num2 == 0)
                {
                    s += "Buzz";
                }
                System.out.println(s);
            }
            else if (i % num2 == 0)
            {
                s += "Buzz";
                System.out.println(s);
            }
            else
            {
                System.out.println(i);
            }
        }
    }
}
