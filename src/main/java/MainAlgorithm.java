
public class MainAlgorithm {

    public void run() {
        SpellObject so = new SpellObject("Lundsgårdsvej");
        SCorrecter.run(so);
        so.print();
    }

    public static void main(String[] args) {
        new MainAlgorithm().run();
    }
}
