import missing_name.graph_connectivity.GraphConnectivityCorrector;
import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import util.SpellObject;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

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
        java.io.File folder = new java.io.File("/home/hannah/workspace/osm_masterthesis/data/atlas");
        java.io.File[] listOfFiles = folder.listFiles();
        assert listOfFiles != null;
        String filename;
        for (int i = 0; i < listOfFiles.length; i++) {
            System.out.println("Processing file " + i);
            filename = listOfFiles[i].getAbsolutePath();
            final File atlasFile = new File(filename);
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
            edges.values().forEach(SpellObject::print);
        }
    }

    public static void main(String[] args) {
        new Main().run();
    }

}
