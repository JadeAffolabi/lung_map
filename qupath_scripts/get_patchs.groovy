import qupath.lib.images.servers.LabeledImageServer
import java.nio.file.Files
import java.nio.file.Paths
import javax.imageio.ImageIO
import javax.imageio.IIOImage
import javax.imageio.ImageWriter
import javax.imageio.ImageWriteParam
import javax.imageio.stream.FileImageOutputStream
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.awt.image.Raster
import qupath.lib.regions.RegionRequest
import groovy.transform.CompileStatic

def imageData = getCurrentImageData()
def server    = imageData.getServer()
def name      = GeneralTools.stripExtension(server.getMetadata().getName())
def pathOutputImages = buildFilePath(PROJECT_BASE_DIR, 'LCHUR', 'imgs')
def pathOutputMasks = buildFilePath(PROJECT_BASE_DIR, 'LCHUR', 'masks')
mkdirs(pathOutputImages)
mkdirs(pathOutputMasks)

// ── Paramètres ─────────────────────────────────────────────────────────────
int    min_space           = 50
double downsample          = 4
int    tileSize            = 2000
double minAnnotatedPercent = 90.0
int    maxTiles            = 4
// ───────────────────────────────────────────────────────────────────────────

// --- Construction du serveur de labels ---
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE)
    .downsample(downsample)
    .addLabel('Tumor',    1)
    .addLabel('Stroma',   2)
    .addLabel('Necrosis', 3)
    .addLabel('Other',    4)
    .multichannelOutput(true)
    .build()

// --- MÉTHODES OPTIMISÉES AVEC @CompileStatic ---

@CompileStatic
void writeTiffMultiband(BufferedImage src, File outFile) {
    Raster srcRaster = src.getRaster()
    int w            = src.getWidth()
    int h            = src.getHeight()
    int numBands     = srcRaster.getNumBands()

    // 1. Extraction ultra-rapide des canaux via getSamples (tout le bloc d'un coup)
    byte[][] bandData = new byte[numBands][w * h]
    for (int band = 0; band < numBands; band++) {
        int[] samples = srcRaster.getSamples(0, 0, w, h, band, (int[]) null)
        byte[] dest = bandData[band]
        for (int i = 0; i < samples.length; i++) {
            dest[i] = (byte) samples[i]
        }
    }

    // 2. Écriture TIFF
    Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("tiff")
    if (!writers.hasNext()) throw new IOException("Aucun writer TIFF disponible")
    ImageWriter writer = writers.next()
    ImageWriteParam param = writer.getDefaultWriteParam()
    FileImageOutputStream stream = new FileImageOutputStream(outFile)
    writer.setOutput(stream)
    writer.prepareWriteSequence(null)

    for (int band = 0; band < numBands; band++) {
        BufferedImage grayImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        byte[] grayData = ((DataBufferByte) grayImg.getRaster().getDataBuffer()).getData()
        System.arraycopy(bandData[band], 0, grayData, 0, grayData.length)

        writer.writeToSequence(new IIOImage(grayImg, null, null), param)
    }

    writer.endWriteSequence()
    stream.close()
    writer.dispose()
}

@CompileStatic
double calculateAnnotatedPercentage(BufferedImage maskImg) {
    Raster raster = maskImg.getRaster()
    int w = maskImg.getWidth()
    int h = maskImg.getHeight()
    int numBands = raster.getNumBands()
    int totalPx = w * h
    
    // Extraction en une seule passe de l'ensemble des pixels entrelacés
    int[] pixels = raster.getPixels(0, 0, w, h, (int[]) null)
    int annotated = 0

    // Parcours du tableau primitif 1D (beaucoup plus rapide que getSample)
    for (int i = 0; i < pixels.length; i += numBands) {
        for (int b = 0; b < numBands; b++) {
            if (pixels[i + b] > 0) {
                annotated++
                break
            }
        }
    }
    
    return (annotated / (double) totalPx) * 100.0
}

// --- Boucle principale ---
int imgWidth  = server.getWidth()
int imgHeight = server.getHeight()
int tileStep  = (int)(tileSize * downsample)

int kept = 0, skipped = 0
print "Analyse et export sélectif des tuiles..."

outer:
for (int y = 0; y < imgHeight; y += tileStep) {
    for (int x = 0; x < imgWidth; x += tileStep) {
        
        if (kept >= maxTiles) {
            print "Limite de ${maxTiles} tuiles atteinte, arrêt."
            break outer
        }
        
        int rWidth  = Math.min(tileStep, imgWidth  - x)
        int rHeight = Math.min(tileStep, imgHeight - y)

        def region = RegionRequest.createInstance(
            labelServer.getPath(), downsample,
            x, y, rWidth, rHeight
        )

        // Lecture du masque
        def maskImg = labelServer.readRegion(region)
        
        // Calcul du pourcentage via la méthode compilée statiquement
        double pct = calculateAnnotatedPercentage(maskImg)

        if (pct < minAnnotatedPercent) {
            skipped++
            continue
        }

        String baseName = String.format("%s_x%d_y%d", name, x, y)

        // Sauvegarde rapide du masque .tif multipage
        writeTiffMultiband(maskImg, new File(pathOutputMasks, baseName + '.tif'))

        // Extraction et sauvegarde de l'image source .png
        def srcRegion = RegionRequest.createInstance(
            server.getPath(), downsample,
            x, y, rWidth, rHeight
        )
        ImageIO.write(server.readRegion(srcRegion), 'png', new File(pathOutputImages, baseName + '.png'))

        kept++
        print "  ✔ ${baseName} — ${String.format('%.1f', pct)}% annotés"
    }
}

print "─── Terminé : ${kept} tuiles exportées, ${skipped} ignorées ───"
