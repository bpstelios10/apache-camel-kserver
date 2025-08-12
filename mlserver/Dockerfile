FROM seldonio/mlserver:1.7.1

# Install the ONNX runtime
RUN pip install mlserver-onnx

# Expose KServe V2 ports
EXPOSE 8080
EXPOSE 8081

# Start MLServer, pointing to model repository
CMD ["mlserver", "start", "/models"]
