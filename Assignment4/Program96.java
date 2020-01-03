import java.util.Scanner;

public class PredictGPAWithWhileLoop {
    public static void main(String[] arg) {
        Scanner scnr = new Scanner(System.in);
        int creditHours;
        int courseGPATimesHours = 0;
        String letterGrade;
        int totalCreditHours = 0;
        int count = 0;

            /*
            My second method where I count the credit hours within the switch: Call it method 2
            I used count to count how many times the while loop would prompt the user.
            I used two counters to keep a running score of how many credit hours the user had
            taken and what the total sum of their GPA * credit hours would be.
            These teo values were used in the final operation of dividing the sum of their GPA * credit hours
            by the total amount of credit hours you were taking.
             */
            while( count < 4)
            {
                count = count + 1;
                System.out.print("Course " + count + ": Number of hours? ");
                creditHours = scnr.nextInt();
                System.out.print("Course " + count + ": Expected grade? ");
                letterGrade = scnr.next();
                switch (letterGrade)
                {
                    case "A":
                        courseGPATimesHours = courseGPATimesHours + (4 * creditHours);

                        break;
                    case "B":
                        courseGPATimesHours = courseGPATimesHours + (3 * creditHours);

                        break;
                    case "C":
                        courseGPATimesHours =courseGPATimesHours + (2 * creditHours);

                        break;
                    case "D":
                        courseGPATimesHours =courseGPATimesHours + (1 * creditHours);

                    case "a":
                        courseGPATimesHours = courseGPATimesHours + (4 * creditHours);

                        break;
                    case "b":
                        courseGPATimesHours =courseGPATimesHours + (3 * creditHours);

                        break;
                    case "c":
                        courseGPATimesHours = courseGPATimesHours + (2 * creditHours);

                        break;
                    case "d":
                        courseGPATimesHours =courseGPATimesHours + (1 * creditHours);

                    default:
                        courseGPATimesHours = 0;
                        break;
                }
//
            totalCreditHours = totalCreditHours + creditHours;

        }
        System.out.print("Your semester GPA would be: " );
        System.out.printf("%.2f", (double)courseGPATimesHours/totalCreditHours);


    }
}
