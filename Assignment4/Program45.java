import java.util.Random;

public class SumRandomIntegers {
    public static void main(String[] args) {
        Random rd = new Random();

        int sum = 0;
        int largest = -1;

        for (int i = 1; i <= 100; i++) {
            int num =  rd.nextInt(1000);
            sum += num;
            if (num > largest)
                largest = num;
        }

        System.out.print("The sum of random 100 integers was ");
        System.out.println(sum);

        System.out.print("The average of 100 random integers was ");
        System.out.println(sum/100.0);

        System.out.print("The largest number among 100 random integers was ");
        System.out.println(largest);
    }
}

