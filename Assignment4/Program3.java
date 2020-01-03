public class Program3 {
    public static void main(String [] args)
    {
        System.out.println("Random Integers ===========");
        int sum = 0;
        int max = 0;
        int temp;

        //for loop
        for (int i = 0; i < 100; i++)
        {
            //generate random int
            temp = (int)(Math.random() * 1000 + 1);
            //add to sum
            sum += temp;
            //find max
            if (temp > max)
            {
                max = temp;
            }
        }

        //calculate average
        double average = (double)sum / 100.0;

        //print
        System.out.println("The sum of 100 random integers was " + sum);
        System.out.println("The average of 100 random integers was " + average);
        System.out.println("The largest number among 100 random integers was " + max);
    }
}
