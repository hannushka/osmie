import missing_name.graph_connectivity.GraphConnectivityCorrector;
import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.names.NameTag;
import spellchecker.random_forest.SCorrector;
import util.SpellObject;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class Main {
    Map<Long, SpellObject> edges;

    public Main() {
        edges = new HashMap<>();
    }

    private void processEdge(SpellObject so, Edge e) {
//        SCorrector.run(so, e);
        GraphConnectivityCorrector.run(so, e);
    }

    public void run() {
        final File atlasFile = new File("data/POINT (7.9321289 55.4665832)=POINT (8.0998535 55.5783983).atlas");
        final Atlas atlasLoad = new AtlasResourceLoader().load(atlasFile);
        SpellObject so;
        for (Edge e : atlasLoad.edges()) {
            long id = Math.abs(e.getIdentifier());
            if (edges.containsKey(id)) {
                so = edges.get(id);
            } else {
                so = new SpellObject(id);
                edges.put(id, so);
            }
            processEdge(so, e);
        }
        edges.values().forEach(spo -> spo.print());
    }

    public static void main(String[] args) {
        new Main().run();
    }
}