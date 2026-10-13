process MULTIQC {
    label 'process_single'

    input:
    path files

    output:
    path "multiqc_report.html", emit: report

    script:
    """
    multiqc .
    """
}
