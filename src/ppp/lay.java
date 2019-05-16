package com.innotechx.dao;//package com.innotechx.dao;
//
//import org.apache.hadoop.conf.Configuration;
//import org.apache.hadoop.hbase.HBaseConfiguration;
//import org.apache.hadoop.hbase.TableName;
//import org.apache.hadoop.hbase.client.*;
//import org.apache.hadoop.hbase.util.Bytes;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.stereotype.Component;
//
//import javax.annotation.PostConstruct;
//import javax.annotation.PreDestroy;
//import java.io.IOException;
//import java.util.NavigableMap;
//
//@Component
//public class HbaseDao {
//
//    private final Logger logger = LoggerFactory.getLogger(this.getClass());
//
//    @Value("${hbase.server}")
//    private String server;
//
//    @Value("${hbase.table}")
//    private String table;
//
//    private Connection conn;
//
//    private HTable htable;
//
//    @PostConstruct
//    public void init() {
//        Configuration configuration = HBaseConfiguration.create();
//        configuration.set("hbase.zookeeper.quorum", server);
//
//        try{
//            conn = ConnectionFactory.createConnection(configuration);
//        } catch (IOException e) {
//            logger.error(e.getMessage());
//        }
//
//        try {
//            htable = (HTable)conn.getTable(TableName.valueOf(table));
//        } catch (IOException e) {
//            logger.error(e.getMessage());
//        }
//    }
//
//    public NavigableMap getRow(long uid, String[] fields) throws IOException {
//        Get get = new Get(Bytes.toBytes(uid));
//        for (String field: fields) {
//            get.addFamily(Bytes.toBytes(field));
//        }
//        Result result = null;
//        result  = htable.get(get);
//        logger.debug("Hbase get row: <uid= " + uid + "> <fields=" + fields.toString() + ">");
//        NavigableMap navigableMap = null;
//        if (result != null) {
//           navigableMap  = result.getFamilyMap(Bytes.toBytes("user"));
//        }
//        return navigableMap;
//    }
//
//    @PreDestroy
//    public void close() {
//        if (this.conn != null) {
//            try {
//                this.conn.close();
//            } catch (Exception e) {
//                logger.error(e.getMessage());
//            }
//        }
//    }
//}
