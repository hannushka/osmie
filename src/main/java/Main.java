import missing_name.graph_connectivity.GraphConnectivityCorrector;
import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.names.NameTag;
import spellchecker.random_forest.SCorrector;
import util.SpellObject;

import java.io.IOException;
import java.util.Optional;

public class Main {


    private void processEdge(Edge e) {
        SpellObject so = new SpellObject(e.getOsmIdentifier());
        SCorrector.run(so, e);
        GraphConnectivityCorrector.run(so, e);
        so.print();
    }

    public void run() {
        final File atlasFile = new File("data/POINT (7.9321289 55.4665832)=POINT (8.0998535 55.5783983).atlas");
        final Atlas atlasLoad = new AtlasResourceLoader().load(atlasFile);
        for (Edge e : atlasLoad.edges()) {
            processEdge(e);
        }
    }

    public static void main(String[] args) {
        new Main().run();
    }
}
