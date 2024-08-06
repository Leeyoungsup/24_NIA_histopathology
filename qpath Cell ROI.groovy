import qupath.lib.gui.scripting.QPEx
import qupath.lib.io.PathIO
import qupath.lib.objects.PathObjects

// 프로젝트의 모든 이미지 처리
def project = QPEx.getProject()

for (entry in project.getImageList()) {
    // 이미지 데이터 불러오기
    def imageData = QPEx.openImageData(entry.getUri())
    QPEx.setCurrentImageData(imageData)

    // 세포 검출 플러그인 실행
    QPEx.runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD", "requestedPixelSizeMicrons": 0.5, "backgroundRadiusMicrons": 8.0, "medianRadiusMicrons": 0.0, "sigmaMicrons": 1.5, "minAreaMicrons": 10.0, "maxAreaMicrons": 400.0, "threshold": 0.1, "cellExpansionMicrons": 5.0, "includeNuclei": true, "smoothBoundaries": true, "makeMeasurements": true}')

    // 검출된 세포 객체 가져오기
    def cells = QPEx.getDetectionObjects().findAll { it.isCell() }
    print 'Detection completed for ' + entry.getImageName() + '! Number of cells detected: ' + cells.size()

    // 내보내기 경로 설정
    def outputPath = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'exported_polygons', entry.getImageName())
    QPEx.mkdirs(outputPath)

    // 다각형 객체 내보내기
    def detections = QPEx.getDetectionObjects()
    def detectionsFile = new File(outputPath, 'detections.geojson')

    // GeoJSON 형식으로 다각형 데이터 내보내기
    PathIO.writeObjectToFile(detectionsFile, detections)

    print 'Export completed for ' + entry.getImageName() + '! Data saved to: ' + detectionsFile.getPath()
}

print 'Batch processing and export completed!'
